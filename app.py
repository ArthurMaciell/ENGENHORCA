# app.py
from dotenv import load_dotenv
load_dotenv() 

import streamlit as st
from pydantic import ValidationError
from typing import Any, Dict

from src.router import route
from src.retrievers import get_tech_retriever, get_product_retriever
from src.chains import build_tech_chain, build_product_chain

# Schemas p/ validar entrada/saída
from src.schemas import QueryRequest, QueryResponse

st.set_page_config(page_title="ENGENHORCA (Tech & Products)", page_icon="🧭", layout="wide")
st.title("🧭 ENGENHORCA — Técnico & Produtos")

with st.sidebar:
    st.header("Configurações")
    k = st.slider("Top-K do retriever", 1, 10, 4)
    mode = st.radio("Modo de Roteamento", ["Auto (Router)", "Forçar Técnico", "Forçar Produto"])
    st.markdown("---")
    st.caption("Coloque os PDFs em `data/tech` e `data/products` e rode `python -m src.ingest`.")

# cacheia recursos pesados (retrievers/chains)
@st.cache_resource
def _build(_k: int):
    tech_ret = get_tech_retriever(k=_k)
    prod_ret = get_product_retriever(k=_k)
    return build_tech_chain(tech_ret), build_product_chain(prod_ret)

tech_chain, prod_chain = _build(k)

# estado do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# render histórico
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# ---- helpers ----
def _to_response(raw: Any, route_hint: str | None = None) -> QueryResponse:
    """
    Normaliza qualquer saída (str/dict) para QueryResponse.
    Mantém retrocompatibilidade se router/chain ainda devolver str.
    """
    if isinstance(raw, dict):
        # tenta mapear campos comuns
        ans = raw.get("answer") or raw.get("output") or raw.get("text") or str(raw)
        return QueryResponse(
            answer=str(ans),
            sources=raw.get("sources"),
            route=raw.get("route", route_hint),
            reason=raw.get("reason"),
        )
    # str ou qualquer outra coisa
    return QueryResponse(answer=str(raw), route=route_hint)

def _render_response(resp: QueryResponse):
    st.markdown(resp.answer)
    meta = []
    if resp.route:
        meta.append(f"**Rota:** `{resp.route}`")
    if resp.reason:
        meta.append(f"**Motivo do roteamento:** {resp.reason}")
    if meta:
        st.markdown("> " + " | ".join(meta))
    if resp.sources:
        with st.expander("Fontes"):
            for s in resp.sources:
                st.markdown(f"- {s}")

# input do usuário
if prompt := st.chat_input("Pergunte algo…"):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # valida entrada
            req = QueryRequest(question=prompt)

            with st.spinner("Pensando…"):
                if mode == "Forçar Técnico":
                    raw = "(forçado: tech)\n\n" + tech_chain.invoke(req.question)
                    resp = _to_response(raw, route_hint="tech")
                elif mode == "Forçar Produto":
                    raw = "(forçado: product)\n\n" + prod_chain.invoke(req.question)
                    resp = _to_response(raw, route_hint="product")
                else:
                    raw = route(req.question, tech_chain, prod_chain)  # pode ser str ou dict
                    resp = _to_response(raw)

            _render_response(resp)
            # salva no histórico apenas o texto final para manter compatibilidade visual
            st.session_state.messages.append(("assistant", resp.answer))

        except ValidationError as ve:
            st.error("Entrada inválida. Detalhes abaixo:")
            st.code(str(ve), language="bash")
            st.session_state.messages.append(("assistant", "Entrada inválida."))
        except Exception as e:
            st.error("Ocorreu um erro ao processar sua pergunta.")
            st.code(repr(e), language="python")
            st.session_state.messages.append(("assistant", "Erro ao processar a pergunta."))
