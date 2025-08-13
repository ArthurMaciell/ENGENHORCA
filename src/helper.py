from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# QA chain (LCEL) — responde usando o retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import os

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyMuPDFLoader)
    
    documents = loader.load()

    return documents


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

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
        "Responda em português.\n"
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