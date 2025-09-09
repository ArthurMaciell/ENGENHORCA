from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

TECH_DIR = "vectorstores/chroma_tech"
PROD_DIR = "vectorstores/chroma_products"

def get_tech_retriever(k: int = 4):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="tech_docs", embedding_function=emb, persist_directory=TECH_DIR)
    return vs.as_retriever(search_kwargs={"k": k})

def get_product_retriever(k: int = 4):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(collection_name="product_docs", embedding_function=emb, persist_directory=PROD_DIR)
    return vs.as_retriever(search_kwargs={"k": k})
