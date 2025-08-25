import os, uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    
def _chunkify(texts, chunk_size=1800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    out = []
    for t in texts:
        out.extend([c.strip() for c in splitter.split_text(t) if c and c.strip()])
    return out

def build_index_from_extracted(extracted, source_name: str):
    # 1) junta tudo que é TEXTO
    base_texts = []
    base_texts.extend(extracted.get("texts", []) or [])
    base_texts.extend(extracted.get("tables", []) or [])
    base_texts.extend(extracted.get("image_text", []) or [])

    # 2) chunking
    chunks = _chunkify(base_texts)

    # 3) vira Documents com metadados
    docs = [Document(page_content=c, metadata={"source": source_name, "kind": "mixed"}) for c in chunks]

    # 4) embeddings + FAISS
    emb = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vs = FAISS.from_documents(docs, emb)

    # 5) retriever (MMR ajuda a diversificar)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.5})
    return retriever
    
    
def build_vectorstore_from_extracted(extracted: dict, source_name: str = "arquivo.pdf", k: int = 3):
    """
    extracted = {
        "texts": [str, ...],
        "tables": [str, ...],
        "image_text": [str, ...]
    }
    """
    all_blocks = []
    for key in ["texts", "tables", "image_text"]:
        for x in extracted.get(key, []):
            if x and str(x).strip():
                all_blocks.append(str(x).strip())

    if not all_blocks:
        raise ValueError("Nada para indexar.")

    # cria Documents direto (sem split extra)
    docs = [Document(page_content=b, metadata={"source": source_name}) for b in all_blocks]

    encoder = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    vs = FAISS.from_documents(docs, encoder)

    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever
