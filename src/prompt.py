# QA chain (LCEL) — responde usando o retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import os

def _format_docs(docs):
    parts = []
    for d in docs:
        src = os.path.basename((d.metadata or {}).get("source", ""))
        page = (d.metadata or {}).get("page")
        header = f"[{src} p.{int(page) if page is not None else '-'}]"
        parts.append(header + "\n" + (d.page_content or ""))
    return "\n\n---\n\n".join(parts)

def build_qa_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        "Resuma em português.\n"
        "Se a resposta não estiver explícita, diga: 'Não encontrei no documento.'\n\n"
        "Pergunta: {question}\n\n"
        "CONTEXTO:\n{context}"
    )
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
