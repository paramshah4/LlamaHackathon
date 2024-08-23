from flask import Flask
from flask_cors import CORS
from flask import request
from copy import deepcopy
import os
from time import time
from dotenv import load_dotenv
import atexit
load_dotenv()
###
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone, ServerlessSpec
###

app = Flask(__name__)
CORS(app)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Get the index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI embedding
embed_model = OpenAIEmbedding()

def delete_pinecone_index():
    print("Deleting Pinecone index...")
    pc.delete_index(PINECONE_INDEX_NAME)
    print("Pinecone index deleted.")

# Register the delete function to be called on exit
atexit.register(delete_pinecone_index)

@app.route("/")
def helloWorld() :
    return "Hello World"

@app.route("/llamaParse", methods=['POST'])
def llamaparse():
    now = time()
    question = request.json['question']
    documents = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown").load_data("uploads/Banana Logistics Historical Financials (2022-12-31).pdf")
    page_nodes = get_page_nodes(documents)
    node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4"), num_workers=8
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    
    # Combine all nodes
    all_nodes = base_nodes + objects + page_nodes
    
    # Embed and upsert nodes to Pinecone
    for i, node in enumerate(all_nodes):
        embedding = embed_model.get_text_embedding(node.get_content())
        index.upsert(vectors=[(str(i), embedding, {"text": node.get_content()})])
    
    # Query embedding
    query_embedding = embed_model.get_text_embedding(question)
    
    # Query Pinecone
    query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    # Extract text from query results
    nodes = [{"text": match['metadata']['text'], "score": match['score']} for match in query_results['matches']]  

    # Use Cohere rerank
    cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=5)
    reranked_nodes = cohere_rerank.postprocess_nodes(nodes)
    reranked_texts = [node['text'] for node in reranked_nodes]
    
    # Use OpenAI to generate final response
    llm = OpenAI(model="gpt-4")
    context = "\n".join(reranked_texts)
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = llm.complete(prompt)

    print(f"Elapsed: {round(time() - now, 2)}s")
    return response.text

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