import os
import streamlit as st
from langchain_groq  import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")


prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}"""
)

def create_vector_embeddings():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyMuPDFLoader("./research_papers/AttentionIsAllYouNeed.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt = st.text_input("<<<ENTER YOUR QUERY FOR THE RESEARCH PAPER>>>")

if st.button("DOCUMENT EMBEDDINGS"):
    create_vector_embeddings()
    st.write("VECTOR DATABASE IS READY!")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input':user_prompt})
    print(f'RESPONSE TIME: [time.process_time() - start]')

    st.write(response['answer'])

