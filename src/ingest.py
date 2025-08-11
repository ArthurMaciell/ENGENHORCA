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

    if debug:
        # contagem por tipo
        def elem_type(e):
            return getattr(e, "category", None) or e.__class__.__name__

        totals = {}
        for e in chunks:
            t = elem_type(e).lower()
            totals[t] = totals.get(t, 0) + 1

        logger.info("Distribuição por tipo:")
        for k, v in sorted(totals.items()):
            logger.info("- %s: %d", k, v)

        # salva prévia
        import os, json
        out = []
        for e in chunks[:10]:
            meta = getattr(e, "metadata", {}) or {}
            out.append({
                "type": elem_type(e),
                "page": meta.get("page_number"),
                "text_preview": (getattr(e, "text", "") or str(e))[:300]
            })
        os.makedirs("data/debug", exist_ok=True)
        with open("data/debug/chunks_preview.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info("Prévia salva em data/debug/chunks_preview.json")
    
    return chunks
