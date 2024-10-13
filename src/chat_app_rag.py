import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def main():
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a FAISS index
    index = faiss.IndexFlatL2(384)  # 384 is the embedding size for MiniLM

    st.title("Document Retrieval with RAG")
    st.write("Upload a text file, and start chatting!")

    # File uploader for text files
    uploaded_file = st.file_uploader("Choose a text file", type="txt")

    # Store embeddings and content for retrieval
    embeddings = []
    file_content = ""

    # Read the content of the uploaded file
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.write("File content loaded!")

        # Split the file content into sentences for better retrieval
        sentences = file_content.split('. ')
        
        # Generate embeddings for each sentence
        sentence_embeddings = model.encode(sentences)
        index.add(np.array(sentence_embeddings).astype('float32'))  # Add embeddings to FAISS index
        embeddings.extend(sentence_embeddings)

        # Show the content of the uploaded file
        if st.checkbox("Show file content"):
            st.write(file_content)

    # User input for querying the document
    user_query = st.text_input("Ask something about the document:")

    if user_query:
        # Generate embedding for the user query
        query_embedding = model.encode([user_query])

        # Search the FAISS index for the nearest neighbors
        D, I = index.search(np.array(query_embedding).astype('float32'), k=3)  # Get top 3 nearest neighbors

        # Retrieve the most similar sentences based on indices
        best_matches = [sentences[idx] for idx in I[0]]  # Get sentences corresponding to indices

        # Display the best matching sentences
        if D[0][0] < 1.0:  # Threshold for similarity (adjust as needed)
            response_text = "Based on your question, here are the most relevant sentences:\n\n" + "\n".join(best_matches)
        else:
            response_text = "Sorry, I couldn't find relevant information based on your query."

        # Display the response
        st.write(response_text)

    # Footer
    st.write("Powered by SentenceTransformers and FAISS")

if __name__ == "__main__":
    main()
