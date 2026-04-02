import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import combine_documents
import time

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PDFPlumberLoader

from langchain_huggingface import HuggingFaceEmbeddings
import tempfile 
from dotenv import load_dotenv

load_dotenv()

# load the groq api key from environment or streamlit secrets
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Use a valid Groq model name
if groq_api_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
else:
    st.error("Please provide a GROQ_API_KEY in Secrets or .env")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector database..."):
            # Handle the uploaded file using a temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Using a lightweight, reliable embedding model
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.loader = PDFPlumberLoader(tmp_path)
            st.session_state.docs = st.session_state.loader.load()
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector database is ready!")

st.title("YOUR DOCS Q & A CHATBOT")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success("File successfully uploaded!")
    if st.button("Document Embedding"):
        create_vector_embedding(uploaded_file)
else:
    st.info("Please upload a PDF file to begin")

user_prompt = st.text_input("Enter your Query")

if user_prompt and "vectors" in st.session_state:
    # Logic is now contained within the session state check to prevent re-runs
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.time()
    response = retriever_chain.invoke({'input': user_prompt})
    st.write(f"Response time: {round(time.time()-start, 2)} seconds")
    
    st.markdown("### Answer")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.divider()
