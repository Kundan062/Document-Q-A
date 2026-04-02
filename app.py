import streamlit as st
import os
import time
import tempfile
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import combine_documents
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)
prompt = ChatPromptTemplate.from_template(
  """
  You are an aviation chat assitant.
  Answer the questions based on the provided context only.
  Please provide the most accurate response based on the question
  <context>
  {context}
  </context>
  Question:{input}
  """
)

def extract_text_with_ocr(pdf_path):
    """Extract text from image-based PDF using OCR"""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        extracted_text = []
        
        for page_num, image in enumerate(images):
            # Use Tesseract OCR to extract text
            text = pytesseract.image_to_string(image)
            extracted_text.append({
                'page': page_num + 1,
                'text': text
            })
        
        return extracted_text
    except Exception as e:
        st.error(f"Error during OCR: {str(e)}")
        return None

def create_vector_embedding(Uploaded_file):
  if "vectors" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
      tmp_file.write(Uploaded_file.read())
      tmp_path = tmp_file.name
    
    st.session_state.embeddings = HuggingFaceEmbeddings()
    
    # First try to extract text using PDFPlumberLoader
    st.session_state.loader = PDFPlumberLoader(tmp_path)
    st.session_state.docs = st.session_state.loader.load()
    
    # If no text extracted, use OCR
    if not st.session_state.docs or all(len(doc.page_content.strip()) == 0 for doc in st.session_state.docs):
        st.info("Detected image-based PDF. Processing with OCR... This may take a moment.")
        ocr_results = extract_text_with_ocr(tmp_path)
        
        if ocr_results:
            # Convert OCR results to Document objects
            st.session_state.docs = [
                Document(page_content=result['text'], metadata={'page': result['page']})
                for result in ocr_results
            ]
        else:
            st.error("Failed to extract text using OCR")
            return
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("YOUR DOCS Q & A CHATBOT")
Uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if Uploaded_file is not None:
    st.success("File successfully uploaded!")
else:
   st.info("Please upload a PDF file to begin")
st.write("Click Document Embedding first")
if st.button("Document Embedding"):
   create_vector_embedding(Uploaded_file)
   st.write("vector database is ready")
if "vectors" in st.session_state:
    user_prompt = st.text_input("enter you Query")
    if user_prompt and st.button("submit"):
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain=create_retrieval_chain(retriever,document_chain)
        start = time.time()
        response = retriever_chain.invoke({'input':user_prompt})
        st.write(f"Response time :{time.time()-start}")
        st.write(response['answer'])
        with st.expander("Document similarity Search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------')
