from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.storage import InMemoryStore  # ou SQLAlchemyStore p/ persistir pais
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from base64 import b64decode
import os
from htmlTemplates import css,bot_template,user_template
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import uuid, hashlib
from io import BytesIO
from typing import List
from __future__ import annotations
from typing import List
from io import BytesIO
import uuid, hashlib

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1800,
        chunk_overlap = 200,
        length_function=len
        
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)

    
    return vectorstore




def get_conversation_chain(vectorstore):
    llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
    )
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                                            memory=memory)
    
    return conversation_chain


def handler_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def chunks_image(pdf_doc):
    chunks = partition_pdf(
        file=pdf_doc,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image","Table"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="basic",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=2000,
        languages=["por","eng"],
        ocr_languages=["por","eng"]

        # extract_images_in_pdf=True,          # deprecated
    )
    
    return chunks