from flask import Flask, request, jsonify, render_template
import os
import faiss
import numpy as np
import cohere
from transformers import BertTokenizer, BertModel
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import numpy as np
import os
import pickle

# Define paths for storing FAISS index and embeddings
EMBEDDINGS_FILE = 'embeddings.npy'
INDEX_FILE = 'faiss_index.bin'
DOCS_FILE = 'documents.pkl'

# Function to generate embeddings for documents and save to disk
def generate_and_save_embeddings(documents):
    embeddings = generate_embeddings(documents)

    # Save embeddings to a file
    np.save(EMBEDDINGS_FILE, embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    # Save FAISS index to a file
    faiss.write_index(faiss_index, INDEX_FILE)

    # Save documents as well (so we can retrieve the correct document by index)
    with open(DOCS_FILE, 'wb') as f:
        pickle.dump(documents, f)

    return embeddings, faiss_index

# Function to load embeddings and FAISS index from disk
def load_embeddings_and_index():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
        # Load embeddings from disk
        embeddings = np.load(EMBEDDINGS_FILE)

        # Load FAISS index from disk
        faiss_index = faiss.read_index(INDEX_FILE)

        # Load documents
        with open(DOCS_FILE, 'rb') as f:
            documents = pickle.load(f)

        return embeddings, faiss_index, documents
    else:
        # If no files are found, return None so that we can regenerate
        return None, None, None


app = Flask(__name__)

# Initialize Cohere client with your API key
COHERE_API_KEY = "7jDhE7Cke10ET0UPPXy7p0GaFqFVSLb7KCYVwyy1"
cohere_client = cohere.Client(COHERE_API_KEY)

# Load the BERT model and tokenizer for generating embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Utility function to read the text from uploaded files
def process_uploaded_files(files):
    documents = []
    for file in files:
        # Read the contents of each file
        content = file.read().decode('utf-8')
        documents.append(content)
    return documents

# Function to generate embeddings for documents
def generate_embeddings(documents):
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use CLS token representation

# Function to retrieve the most similar documents using FAISS
def retrieve_documents(query, embeddings, documents, k=2):
    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]

    # Load the FAISS index (already loaded in previous steps)
    faiss_index = faiss.read_index(INDEX_FILE)

    # Search for top-k most similar documents
    D, I = faiss_index.search(np.array([query_embedding]), k)
    retrieved_documents = [documents[idx] for idx in I[0]]  # Return top-k documents
    return retrieved_documents

# Function to generate a response using Cohere based on retrieved documents
def generate_response(query, retrieved_documents):
    retrieved_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_documents)])

    prompt = f"""
    User Query: {query}
    Relevant Documents:
    {retrieved_text}
    
    Based strictly on the information in the retrieved documents, provide a direct answer to the user's query.
    """

    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=150,
        temperature=0.5
    )
    
    return response.generations[0].text

# Route to serve the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Handle Q&A with uploaded files
@app.route('/qa', methods=['POST'])
def qa():
    query = request.form.get('query')
    uploaded_files = request.files.getlist('files')

    if not query or not uploaded_files:
        return jsonify({"error": "Query and files are required."}), 400

    # Process uploaded files and get document content
    documents = process_uploaded_files(uploaded_files)

    # Try to load the embeddings and FAISS index from disk
    embeddings, faiss_index, saved_documents = load_embeddings_and_index()

    if embeddings is None:
        # If not found, generate embeddings and save them
        embeddings, faiss_index = generate_and_save_embeddings(documents)

    # Retrieve relevant documents using FAISS
    retrieved_documents = retrieve_documents(query, embeddings, documents, k=2)

    # Generate a response based on the retrieved documents
    answer = generate_response(query, retrieved_documents)
    
    return jsonify({"query": query, "answer": answer})


if __name__ == '__main__':
    app.run(debug=True)
