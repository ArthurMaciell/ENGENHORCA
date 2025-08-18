from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from base64 import b64decode
import os

# LLM - Groq com LLaMA 3
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# Separa textos e imagens, mas ignora imagens no processamento
def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)  # ← será ignorado, mas salvo se quiser usar depois
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def _to_text(elem):
    # Se for Document do LangChain
    if hasattr(elem, "page_content"):
        return elem.page_content
    # Se tiver atributo .text
    if hasattr(elem, "text"):
        return elem.text
    # Se for dict
    if isinstance(elem, dict):
        return elem.get("page_content") or elem.get("text") or str(elem)
    # Se for string
    return str(elem)

# Monta prompt apenas com texto extraído
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text + "\n\n"

    prompt_template = f"""
    Você é um assistente técnico. Responda com base apenas no seguinte contexto (extraído de documentos PDF e imagens convertidas para texto):

    {context_text}

    Pergunta: {user_question}
    Resposta:
        """

    return prompt_template.strip()

# Chain simples
def chain(retriever):
    return (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
        )

# Chain com fontes (retorna também o contexto usado)
def chain_with_sources(retriever):
    return {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | model
            | StrOutputParser()
        )
    )