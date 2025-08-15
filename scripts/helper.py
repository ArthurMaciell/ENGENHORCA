from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from unstructured.documents.elements import Table, Image
from io import BytesIO
import base64
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pytesseract
from PIL import Image as PILImage  # evita conflito de nomes
from unstructured.documents.elements import (
    Table,
    Image as USImage,
    CompositeElement,
)


def get_pdf_text(pdf_docs,sep_between_pdfs=True):
    pieces = [] 
    for i, pdf in enumerate(pdf_docs):
        reader = PdfReader(pdf)
        for page in reader.pages:
            pieces.append(page.extract_text() or "")  # evita None
        if sep_between_pdfs and i < len(pdf_docs) - 1:
            pieces.append("\n" + ("—" * 20) + "\n")   # separador
    return "".join(pieces)


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

    
    return vectorstore




def get_conversation_chain(vectorstore):
    llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
    )
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # diversifica
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
        )
    
    # 1. Retriever que leva em conta histórico
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever,
        ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])
    )

    # 2. Prompt de resposta com contexto
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente especializado. Use o contexto recuperado para responder."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    # 3. Cria a chain principal
    retrieval_chain = create_retrieval_chain(history_aware_retriever, llm, prompt)

    # 4. Adiciona memória de sessão (equivalente ao ConversationBufferMemory)
    session_histories = {}

    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: session_histories.setdefault(session_id, ChatMessageHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return chain_with_history


# ✅ Para o caso "Não" (ConversationalRetrievalChain: precisa de dict {"question": ...})
def handler_user_input(user_question: str):
    import streamlit as st

    conv = st.session_state.get("conversation")
    if conv is None:
        st.error("Chain não inicializada. Clique em 'Chunks' primeiro.")
        return

    # ConversationalRetrievalChain espera dict:
    resp = conv.invoke({"question": user_question})   # <<--- chave "question"

    # A resposta típica desse chain:
    # resp = {"answer": "...", "source_documents": [...]}
    answer = resp.get("answer", "")

    # Recupera o histórico diretamente da memória do chain:
    try:
        messages = conv.memory.chat_memory.messages  # lista de mensagens (AI/Human)
    except Exception:
        messages = []

    # Renderiza
    for i, message in enumerate(messages):
        if getattr(message, "type", "").lower() in ("human", "user") or i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # também mostra a última resposta (garante que aparece mesmo se memória não atualizou ainda)
    if answer:
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

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

def extract_elements(chunks):
    tables, texts, images_b64 = [], [], []

    for ch in chunks:
        # Tabela “pura”
        if isinstance(ch, Table):
            tables.append(ch)

        # Tabelas/Imagens aninhadas
        if hasattr(ch.metadata, "orig_elements") and ch.metadata.orig_elements:
            for el in ch.metadata.orig_elements:
                if isinstance(el, Table):
                    tables.append(el)
                # >>> AQUI: checar Image do unstructured, não PIL
                if isinstance(el, USImage) and getattr(el.metadata, "image_base64", None):
                    images_b64.append(el.metadata.image_base64)

        # Texto (CompositeElement)
        if isinstance(ch, CompositeElement):
            # troque para texts.append(ch) se você QUISER o objeto
            texts.append(getattr(ch, "text", ""))

        # (Opcional) imagem “pura” no nível do chunk
        if isinstance(ch, USImage) and getattr(ch.metadata, "image_base64", None):
            images_b64.append(ch.metadata.image_base64)
            
    print(f'O número de chunks é: {len(chunks)}')
    print(f'O número de tabelas é: {len(tables)}')
    print(f'O número de textos é: {len(texts)}')
    print(f'O número de imagens é: {len(images_b64)}')

    return tables, texts, images_b64
            
def ocr_from_images_base64(images_b64):
    image_texts = []
    for b64 in images_b64:
        try:
            image_data = base64.b64decode(b64)
            image = PILImage.open(BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang="por+eng")
            image_texts.append(text)
        except Exception as e:
            print(f"❌ Erro ao processar imagem: {e}")
            image_texts.append(text.strip())
    return image_texts

    
def save_uploaded_file(up, base_dir="data/raw"):
    os.makedirs(base_dir, exist_ok=True)
    # cuidado com nomes repetidos; aqui uso o nome original:
    path = os.path.join(base_dir, up.name)
    with open(path, "wb") as f:
        f.write(up.getbuffer())
    return path  # << string


# ✅ Para o caso "Sim" (sua LCEL chain `chain(retriever)`: retorna string, sem memória)
def handler_user_input_image(user_question: str):

    conv = st.session_state.get("conversation")
    if conv is None:
        st.error("Chain não inicializada. Clique em 'Chunks' primeiro.")
        return

    # sua LCEL chain aceita string e retorna string:
    answer = conv.invoke(user_question)   # <<--- string de saída

    # mantenha um histórico simples no Streamlit (como lista de tuplas, por exemplo)
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("ai", answer))

    # Renderiza
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", text), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", text), unsafe_allow_html=True)
