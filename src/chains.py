# src/chains.py
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .prompts import tech_prompt, product_prompt

def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY não encontrado. Defina no .env ou em st.secrets "
            "(Settings → Secrets) e reinicie o app."
        )
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)

def _format_docs(docs):
    if not docs:
        return "—"
    return "\n\n".join([f"- {d.page_content}" for d in docs])

def build_tech_chain(tech_retriever):
    llm = _get_llm()
    
    chain = (
        {"question": RunnablePassthrough(), "context": tech_retriever | RunnableLambda(_format_docs)}
        | tech_prompt
        | llm
        | StrOutputParser()
    )
    return chain.with_config(
        tags=["ENGENHORCA", "chain:tech"],
        metadata={"component": "tech_chain", "model": "llama-3.1-8b-instant"}
    )

def build_product_chain(product_retriever):
    llm = _get_llm()
    chain = (
        {"question": RunnablePassthrough(), "context": product_retriever | RunnableLambda(_format_docs)}
        | product_prompt
        | llm
        | StrOutputParser()
    )
    return chain.with_config(
        tags=["ENGENHORCA", "chain:product"],
        metadata={"component": "product_chain", "model": "llama-3.1-8b-instant"}
    )
