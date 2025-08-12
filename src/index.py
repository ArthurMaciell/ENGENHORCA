import os, uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

def build_vectorstore(cfg):
    emb = HuggingFaceEmbeddings(model_name=cfg["embeddings"]["model"])
    vs = Chroma(
        collection_name=cfg["retriever"]["collection_name"],
        embedding_function=emb,
        persist_directory=cfg["paths"]["chroma_dir"],
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vs, docstore=store, id_key="doc_id"
    )
    return retriever, vs

def add_documents(retriever, vs, extracted, doc_path, summaries):
    # doc_id por arquivo para amarrar filhos → pai
    doc_id = str(uuid.uuid4())
    # filhos: textos resumidos (ou brutos), tabelas (html), OCRs
    children = []

    # textos
    texts = summaries["texts"] if summaries else extracted["texts"]
    for t in texts:
        children.append(Document(page_content=t, metadata={"doc_id": doc_id, "source": doc_path, "type":"text"}))

    # tabelas
    #for thtml in extracted["tables_html"]:
        #children.append(Document(page_content=thtml, metadata={"doc_id": doc_id, "source": doc_path, "type":"table"}))

    # OCR
    for o in extracted["image_text"]:
        children.append(Document(page_content=o, metadata={"doc_id": doc_id, "source": doc_path, "type":"ocr"}))

    if children:
        retriever.vectorstore.add_documents(children)

    # pai (original bruto) — útil se quiser recuperar “inteiros”
    retriever.docstore.mset([(doc_id, {"path": doc_path})])
    vs.persist()
    
    print('iHAA')



    