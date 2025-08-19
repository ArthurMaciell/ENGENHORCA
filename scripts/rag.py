from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from base64 import b64decode
import os
import base64

# LLM - Groq com LLaMA 3
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)

def is_base64(data: str) -> bool:
    try:
        # tenta decodificar e recodificar
        return base64.b64encode(base64.b64decode(data)) == data.encode()
    except Exception:
        return False

# Separa textos e imagens, mas ignora imagens no processamento
def parse_docs(docs):
    images_b64, texts_norm = [], []
    for doc in docs:
        if is_base64(doc):
            images_b64.append(doc)
        else:
            texts_norm.append(_to_text(doc))  # <- normaliza aqui
    return {"images": images_b64, "texts": texts_norm}

def _to_text(elem):
    # LangChain Document
    if hasattr(elem, "page_content"):
        return elem.page_content
    # Unstructured element
    if hasattr(elem, "text"):
        return elem.text
    # dict com possíveis chaves
    if isinstance(elem, dict):
        return elem.get("page_content") or elem.get("text") or str(elem)
    # fallback
    return str(elem)


# Monta prompt apenas com texto extraído
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # como parse_docs já normaliza, aqui é só juntar
    texts = docs_by_type.get("texts", [])
    # (opcional) limitar tamanho total do contexto p/ evitar prompt gigante
    context_text = "\n\n".join(texts)[:120000]  # ~120k chars de teto

    prompt_template = f"""
    Você é um assistente técnico. Responda **apenas** com base no contexto abaixo (extraído de PDFs e imagens OCR):

    {context_text}

    Pergunta: {user_question}
    Responda de forma direta, cite trechos relevantes quando necessário e diga “Não encontrado no contexto” se faltar informação.
    """.strip()

    return prompt_template

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