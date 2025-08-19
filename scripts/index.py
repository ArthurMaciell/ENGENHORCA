import os, uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

def build_vectorstore():
    # Melhor para PT-BR:
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=emb,
        persist_directory="data/chroma",
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vs, docstore=store, id_key="doc_id"
    )
    return retriever, vs

ID_KEY = "doc_id"

def _norm_list(xs):
    out = []
    for x in xs or []:
        s = getattr(x, "text", x)  # aceita Element do unstructured ou str
        if s is None:
            continue
        s = str(s).strip()
        if s:
            out.append(s)
    return out

def add_documents(retriever, vs, extracted, doc_path, summaries):
    # 1) Coleta bruto e resumos por tipo
    texts  = _norm_list(extracted.get("texts"))
    tables = _norm_list(extracted.get("tables"))
    images = _norm_list(extracted.get("image_text"))   # OCR

    text_summaries  = _norm_list((summaries or {}).get("texts"))
    table_summaries = _norm_list((summaries or {}).get("tables"))
    image_summaries = _norm_list((summaries or {}).get("images"))

    # 2) TEXTS — filhos (resumos) no vectorstore, pais (bruto) no docstore
    if texts:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=text_summaries[i], metadata={ID_KEY: doc_ids[i]})
            for i in range(min(len(text_summaries), len(doc_ids)))
        ]
        if summary_texts:
            retriever.vectorstore.add_documents(filter_complex_metadata(summary_texts))
        # pais
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # 3) TABLES — (igual ao seu notebook: só pais no docstore; filhos opcional)
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=table_summaries[i], metadata={ID_KEY: table_ids[i]})
            for i in range(min(len(table_summaries), len(table_ids)))
        ]
        # se quiser indexar resumos de tabela também, descomente:
        # if summary_tables:
        #     retriever.vectorstore.add_documents(filter_complex_metadata(summary_tables))
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # 4) IMAGES (OCR) — filhos (resumos) no vectorstore, pais (bruto OCR) no docstore
    if images:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=image_summaries[i], metadata={ID_KEY: img_ids[i]})
            for i in range(min(len(image_summaries), len(img_ids)))
        ]
        if summary_img:
            retriever.vectorstore.add_documents(filter_complex_metadata(summary_img))
        retriever.docstore.mset(list(zip(img_ids, images)))

    # 5) Persistir Chroma (se estiver usando persist_directory)
    try:
        vs.persist()
    except Exception:
        pass
    



    