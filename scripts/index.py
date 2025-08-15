import os, uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

def build_vectorstore():
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

def add_documents(retriever, vs, extracted, doc_path, summaries):
    doc_id = str(uuid.uuid4())

    # garanta string:
    if hasattr(doc_path, "name") and hasattr(doc_path, "getvalue"):  # UploadedFile
        source_str = doc_path.name
    else:
        source_str = str(doc_path)

    children = []
    texts = (summaries or {}).get("texts") or extracted.get("texts", [])
    for t in texts:
        children.append(Document(page_content=t, metadata={"doc_id": doc_id, "source": source_str, "type":"text"}))

    for o in extracted.get("image_text", []):
        children.append(Document(page_content=o, metadata={"doc_id": doc_id, "source": source_str, "type":"ocr"}))

    if children:
        # segurança extra: filtra metadados complexos se algo escapar
        from langchain_community.vectorstores.utils import filter_complex_metadata
        children = filter_complex_metadata(children)
        retriever.vectorstore.add_documents(children)

    retriever.docstore.mset([(doc_id, {"path": source_str})])  # << só string aqui
    vs.persist()
    return doc_id



    