import streamlit as st
import cohere

def main():
    # Initialize Cohere API
    COHERE_API_KEY = "7jDhE7Cke10ET0UPPXy7p0GaFqFVSLb7KCYVwyy1"  # Replace with your actual Cohere API key
    co = cohere.Client(COHERE_API_KEY)

    # Streamlit UI
    st.title("Chat with Your Document")
    st.write("Upload a text file, and start chatting!")

    # File uploader for text files
    uploaded_file = st.file_uploader("Choose a text file", type="txt")

    # Function to read the content of the uploaded file
    def read_file(file):
        if file is not None:
            return file.read().decode("utf-8")
        return None

    # When a file is uploaded
    if uploaded_file:
        file_content = read_file(uploaded_file)
        st.write("File content loaded!")

        # Optional: Show the content of the uploaded file
        if st.checkbox("Show file content"):
            st.write(file_content)

        # User input for querying the document
        user_query = st.text_input("Ask something about the document:")

        if user_query:
            # Generate a response using Cohere's language model
            response = co.generate(
                model='command-xlarge-nightly',  # Use an appropriate model from Cohere
                prompt=f"File content: {file_content}\nUser query: {user_query}\nAnswer:",
                max_tokens=100
            )
            
            # Display the response
            st.write(response.generations[0].text.strip())

    # Footer
    st.write("Powered by Cohere")

if __name__ == "__main__":
    main()
