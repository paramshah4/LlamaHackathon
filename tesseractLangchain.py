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
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import MarkdownElementNodeParser

app = Flask(__name__)
CORS(app)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

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
    
    split_documents = [TextNode(text=chunk) for chunk in split_texts]
    page_nodes = get_page_nodes(split_documents)

    node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-4o"), num_workers=8)
    nodes = node_parser.get_nodes_from_documents(split_documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    recursive_index = VectorStoreIndex(nodes=base_nodes + objects + page_nodes)
    cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=5)
    recursive_query_engine = recursive_index.as_query_engine(
        llm=OpenAI(model="gpt-4o"),
        similarity_top_k=5, 
        node_postprocessors=[cohere_rerank],
        verbose=True
    )
    
    response = recursive_query_engine.query(question)
    print(f"Elapsed: {round(time() - now, 2)}s")
    return response.response

def get_page_nodes(docs, separator="\n---\n"):
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)
    return nodes

if __name__ == '__main__':
    app.run(debug=True)
