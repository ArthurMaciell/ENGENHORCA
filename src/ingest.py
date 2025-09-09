import os, glob
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

TECH_DIR = "vectorstores/chroma_tech"
PROD_DIR = "vectorstores/chroma_products"

def load_docs_from(folder: str):
    docs = []
    for path in glob.glob(os.path.join(folder, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        try:
            if path.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
            else:
                docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"[WARN] falha ao ler {path}: {e}")
    return docs

def build_index(data_folder: str, persist_dir: str, collection_name: str):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name=collection_name, embedding_function=emb, persist_directory=persist_dir)
    docs = load_docs_from(data_folder)
    if not docs:
        print(f"Nenhum documento em {data_folder}")
        return
    vs.add_documents(docs)
    vs.persist()
    print(f"Indexados {len(docs)} docs em {persist_dir}")

if __name__ == "__main__":
    load_dotenv()
    build_index("data/tech", TECH_DIR, "tech_docs")
    build_index("data/products", PROD_DIR, "product_docs")
