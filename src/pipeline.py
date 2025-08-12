import os, json
from config import load_settings
from extract import tables_to_html,extract_elements,ocr_from_images_base64
from summarize import build_summarizer, safe_batch
from index import build_vectorstore, add_documents
from ingest import load_pdf_chunks
    
    
if __name__ == '__main__':
    output_path = "./data/raw/"
    path = output_path + 'Sirocco - Linha Conforto.pdf'
    chunks = load_pdf_chunks(path)
    
    tables, texts, images_b64 = extract_elements(chunks)

    image_text = ocr_from_images_base64(images_b64)
    
    cfg = load_settings()
    summarize_chain = build_summarizer(cfg)
    text_summaries = safe_batch(summarize_chain, texts)
    print(text_summaries)
    table_summaries = safe_batch(summarize_chain, tables)
    image_summaries = safe_batch(summarize_chain, image_text)
    
    retriever, vs = build_vectorstore(cfg)
    extracted = {
    "texts": texts,                   # lista de strings (ou elem.text)       
    "image_text": image_text,           # lista de strings vindas do OCR
    "images": image_summaries,        # lista de resumos textuais das imagens (opcional)
        }
    summaries = {
        "texts": [s for s in (text_summaries + table_summaries + image_summaries) if s and s.strip()]
    }
    add_documents(retriever, vs, extracted, path,summaries)