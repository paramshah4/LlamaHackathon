from flask import Flask
from flask_cors import CORS
from flask import request
from copy import deepcopy
import os
from time import time
from dotenv import load_dotenv
load_dotenv()
import atexit
import pinecone
import cohere
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings, VectorStoreIndex
from pinecone import Pinecone, ServerlessSpec
import pytesseract
import pdf2image

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
    if PINECONE_INDEX_NAME in pinecone.list_indexes():
        pc.delete_index(PINECONE_INDEX_NAME)

# Register the delete function to be called on exit
atexit.register(delete_pinecone_index)

# Create the index when the application starts
create_pinecone_index()

@app.route("/")
def helloWorld() :
    return "Hello World"

@app.route("/tesseract", methods=['POST'])
def tesseract():
    now = time()
    question = request.json['question']

    # Tessearct OCR
    pdf_path = "uploads/Banana Logistics Historical Financials (2022-12-31).pdf"
    images = pdf2image.convert_from_path(pdf_path)
    
    documents = []
    for image in images:
        text = pytesseract.image_to_string(image)
        documents.append(TextNode(text=text))
    
    page_nodes = get_page_nodes(documents)
    node_parser = MarkdownElementNodeParser(
        llm=OpenAI(model="gpt-4"), num_workers=8
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    all_nodes = base_nodes + objects + page_nodes

    # Initialize Pinecone vector store
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Configure LlamaIndex settings
    Settings.llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

    # Create index with Pinecone vector store
    pinecone_index = VectorStoreIndex.from_vector_store(
        vector_store,
        nodes=all_nodes
    )

    cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=5)
    pinecone_query_engine = pinecone_index.as_query_engine(
        similarity_top_k=5, node_postprocessors=[cohere_rerank], verbose=True
    )

    response = pinecone_query_engine.query(question)
    print(f"Elapsed: {round(time() - now, 2)}s")
    return response.response

def get_page_nodes(docs, separator="\n---\n"):
    """Split each document into page node, by separator."""
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