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
    print(len(chunks))
    
    image_text = ocr_from_images_base64(images_b64)