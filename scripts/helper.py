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

def chunks_image(pdf_docs):
    """
    pdf_docs: lista de streamlit.UploadedFile (st.file_uploader(..., accept_multiple_files=True))
    Retorna uma lista de chunks do Unstructured.
    """
    all_chunks = []

    if not pdf_docs:
        return all_chunks

    for uf in pdf_docs:
        # Garante o ponteiro no início e extrai bytes
        uf.seek(0)
        pdf_bytes = uf.read()

        # Passa o arquivo como bytes (file=...), e usa metadata_filename só para registrar o nome
        elements = partition_pdf(
            file=BytesIO(pdf_bytes),
            strategy="hi_res",                    # use "ocr_only" se não tiver dependências do hi_res
            infer_table_structure=True,
            metadata_filename=uf.name,
        )

        # Chunking (ajuste os parâmetros conforme seu projeto)
        chunks = chunk_by_title(
            elements,
            extract_image_block_types=["Image","Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="basic",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=2000,
            languages=["por","eng"],
            ocr_languages=["por","eng"]
        )
        all_chunks.extend(chunks)

    return all_chunks


def _hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def chunks_image_1(pdf_names: List[str], pdf_bytes_list: List[bytes]):
    """
    Use cache para não reprocessar tudo a cada rerun do Streamlit.
    Passe listas paralelas: nomes e bytes (mesmo índice).
    """
    all_chunks = []
    seen = set()  # (doc_id, text_hash)

    for name, pdf_bytes in zip(pdf_names, pdf_bytes_list):
        doc_id = str(uuid.uuid4())

        elements = partition_pdf(
            file=BytesIO(pdf_bytes),
            strategy="hi_res",            # troque se precisar: "fast" | "ocr_only"
            infer_table_structure=True,
            metadata_filename=name,
        )

        file_chunks = chunk_by_title(
            elements,
            max_characters=1000,
            combine_text_under_n_chars=200,
            new_after_n_chars=800,
        )

        # anexa metadados e evita duplicatas
        for idx, ch in enumerate(file_chunks):
            # garante que existe metadata dict
            md = getattr(ch, "metadata", None)
            if md is None:
                ch.metadata = {}
                md = ch.metadata

            md["doc_id"] = doc_id
            md["source_name"] = name
            md["chunk_index"] = idx

            text = getattr(ch, "text", "") or ""
            key = (doc_id, _hash_text(text))
            if key in seen:
                continue
            seen.add(key)
            all_chunks.append(ch)

    # debug opcional
    st.write(f"Arquivos processados: {len(pdf_names)}")
    st.write(f"Total de chunks: {len(all_chunks)}")

    return all_chunks


        