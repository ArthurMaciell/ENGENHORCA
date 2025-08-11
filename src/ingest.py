from unstructured.partition.pdf import partition_pdf
import logging
import os, logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger('engenhorca.ingest')


def load_pdf_chunks(file_path: str, debug: bool = True):


    
    # Atenção: garanta que data/raw existe
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="basic",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
        languages=["por", "eng"],
        ocr_languages=["por", "eng"],
    )
    
    logger.info("PDF: %s | chunks gerados: %d", file_path, len(chunks))


    
    return chunks
