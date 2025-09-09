from dotenv import load_dotenv
load_dotenv()  # <- antes de qualquer import que usa a variável
from src.router import route

import streamlit as st
from src.retrievers import get_tech_retriever, get_product_retriever
from src.chains import build_tech_chain, build_product_chain

st.set_page_config(page_title="RAG Router (Tech & Products)", page_icon="🧭", layout="wide")
st.title("🧭 ENGENHORCA - RAG Router — Técnico & Produtos")

with st.sidebar:
    st.header("Configurações")
    k = st.slider("Top-K do retriever", 1, 10, 4)
    mode = st.radio("Modo de Roteamento", ["Auto (Router)", "Forçar Técnico", "Forçar Produto"])
    st.markdown("---")
    st.caption("Coloque seus PDFs em `data/tech` e `data/products` e rode `python -m src.ingest`.")

# cria retrievers e chains (cacheia para não recarregar a cada interação)
@st.cache_resource
def _build():
    tech_ret = get_tech_retriever(k=k)
    prod_ret = get_product_retriever(k=k)
    return build_tech_chain(tech_ret), build_product_chain(prod_ret)

tech_chain, prod_chain = _build()

# estado do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# render histórico
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# input do usuário
if prompt := st.chat_input("Pergunte algo…"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            if mode == "Forçar Técnico":
                answer = "(forçado: tech)\n\n" + tech_chain.invoke(prompt)
            elif mode == "Forçar Produto":
                answer = "(forçado: product)\n\n" + prod_chain.invoke(prompt)
            else:
                answer = route(prompt, tech_chain, prod_chain)
        st.markdown(answer)
    st.session_state.messages.append(("assistant", answer))
