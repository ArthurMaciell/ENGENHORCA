from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image
from io import BytesIO
import base64
from unstructured.documents.elements import (
    Table,
    Image as USImage,
    CompositeElement,
)
import pytesseract
from PIL import Image as PILImage  # evita conflito de nomes
import os
import streamlit as st
from htmlTemplates import css,bot_template,user_template
from langchain_core.messages import HumanMessage, AIMessage

# Pegando o texto dos PDFs
def get_pdf_text(pdf_docs,sep_between_pdfs=True):
    pieces = [] 
    for i, pdf in enumerate(pdf_docs):
        reader = PdfReader(pdf)
        for page in reader.pages:
            pieces.append(page.extract_text() or "")  # evita None
        if sep_between_pdfs and i < len(pdf_docs) - 1:
            pieces.append("\n" + ("—" * 20) + "\n")   # separador
    return "".join(pieces)

# Transformando os textos em chunks
def get_text_chunks(text, chunk_size=1800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],  # tenta do maior pro menor
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
        )
    return splitter.split_text(text or "")

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("Nenhum chunk de texto disponível para indexar (text_chunks está vazio).")

    
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    
    #Para ter memória entre as execuções
    #vectorstore.save_local("faiss_index_dir")
    # e depois:
    #FAISS.load_local("faiss_index_dir", embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_kwargs={'k':3})
    
    return retriever

def format_docs(docs):
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def get_conversation_chain_rag(retriever):
    llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Você é um assistente HVAC. Responda **apenas** com base no contexto.\n\n"
        "Contexto:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    chain = {
        "question": RunnablePassthrough(),                       # passa a pergunta
        "history": itemgetter("history"),                        # se você tiver histórico
        "context": itemgetter("question")                        # usa a pergunta
                | retriever                                   # busca no índice
                | RunnableLambda(format_docs),                # formata os docs
    } | prompt | llm | StrOutputParser()

    return chain

# ✅ Para o caso "Não" (ConversationalRetrievalChain: precisa de dict {"question": ...})
def handler_user_input_simples(user_question: str,retriever):
    # garante que a conversa e o histórico existem
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain_rag(retriever)  # sua função que retorna (prompt | llm | parser)
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    chain = st.session_state.conversation
    history = st.session_state.chat_history
    # CHAME com .invoke e passe o histórico
    answer = chain.invoke({"question": user_question, "history": history})

    # atualiza histórico (seu loop assume pares Human/AI alternados)
    st.session_state.chat_history = history + [
        HumanMessage(content=user_question),
        AIMessage(content=answer),
    ]

    # renderização (igual à sua, só garantindo .content)
    for i, message in enumerate(st.session_state.chat_history):
        content = getattr(message, "content", str(message))
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)


