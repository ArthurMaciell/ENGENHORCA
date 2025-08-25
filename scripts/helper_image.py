from unstructured.partition.pdf import partition_pdf
import pytesseract
from unstructured.documents.elements import (
    Table,
    Image as USImage,
    CompositeElement,
)
import base64
from PIL import Image as PILImage
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from htmlTemplates import css,bot_template,user_template



def format_docs(docs):
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def chunks_image(pdf_doc):
    chunks = partition_pdf(
        file=pdf_doc,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image","Table"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="basic",          # or 'basic'
        max_characters=5000,                  # defaults to 500
        combine_text_under_n_chars=500,       # defaults to 0
        new_after_n_chars=500,
        languages=["por","eng"]

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


def get_conversation_chain_image_rag(retriever):
    llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Você é um assistente HVAC. Dê a melhor explicação baseada no documento lido.  \n\n"
        "Contexto:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    chain = {
        "question": RunnablePassthrough(),                       # passa a pergunta
        "history": itemgetter("history"),                        # se você tiver histórico
        "context": itemgetter("question")                       # usa a pergunta
                | retriever                                   # busca no índice
                | RunnableLambda(format_docs),                # formata os docs
    } | prompt | llm | StrOutputParser()
    
    return chain

# ✅ Para o caso "Sim"
def handler_user_input_image_rag(user_question: str, retriever):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain_image_rag(retriever)
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    chain = st.session_state.conversation
    history = st.session_state.chat_history[-20:]

    raw = chain.invoke({"question": user_question, "history": history})

    # 🔧 Normaliza a resposta para string
    if isinstance(raw, dict):
        answer = raw.get("answer") or raw.get("response") or ""
        # fallback final (evita mostrar dict bruto)
        if not isinstance(answer, str):
            import json
            answer = json.dumps(raw, ensure_ascii=False)
    else:
        answer = raw  # já é string

    st.session_state.chat_history = history + [
        HumanMessage(content=user_question),
        AIMessage(content=answer),
    ]

    for i, message in enumerate(st.session_state.chat_history):
        content = getattr(message, "content", str(message))
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)