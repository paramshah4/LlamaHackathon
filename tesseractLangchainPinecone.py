from flask import Flask, request
from flask_cors import CORS
import os
from time import time
import pdf2image
import pytesseract
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import MarkdownElementNodeParser
from pinecone import Pinecone, ServerlessSpec
import atexit
import cohere

app = Flask(__name__)
CORS(app)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Initialize Pinecone and Cohere
pc = Pinecone(api_key=PINECONE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

# Create Pinecone index
def create_pinecone_index():
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

# Delete Pinecone index
def delete_pinecone_index():
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(PINECONE_INDEX_NAME)

# Register the delete function to be called on exit
atexit.register(delete_pinecone_index)

# Create the index when the application starts
create_pinecone_index()

# LangChain
def split_document(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

@app.route("/tesseractLangchain", methods=['POST'])
def tesseractLangchain():
    now = time()
    question = request.json['question']
    
    pdf_path = "uploads/Banana Logistics Historical Financials (2022-12-31).pdf"
    images = pdf2image.convert_from_path(pdf_path)
    documents = []
    
    for image in images:
        text = pytesseract.image_to_string(image)
        documents.append(TextNode(text=text))
    
    # langchain
    combined_text = " ".join([doc.text for doc in documents])
    split_texts = split_document(combined_text)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create Pinecone vector store
    index = pc.Index(PINECONE_INDEX_NAME)
    vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
    
    # Add texts to the vectorstore
    vectorstore.add_texts(split_texts)
    
    # Perform similarity search
    similar_docs = vectorstore.similarity_search(question, k=5)

    print(f"Number of similar documents found: {len(similar_docs)}")
    
    if not similar_docs:
        return "No relevant documents found to answer the question."
    
    # Apply Cohere rerank
    rerank_results = co.rerank(
        query=question,
        documents=[doc.page_content for doc in similar_docs],
        top_n=5
    )
    
    # Extract reranked documents
    reranked_docs = [similar_docs[i] for i, result in enumerate(rerank_results) if i < len(similar_docs)]    

    # Use OpenAI to generate the final response
    llm = OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    context = "\n".join([doc.page_content for doc in reranked_docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = llm.predict(prompt)
    
    print(f"Elapsed: {round(time() - now, 2)}s")
    return response

if __name__ == '__main__':
    app.run(debug=True)