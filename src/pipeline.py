import os, json
from .config import load_settings
from .extract import extract_from_pdf, file_hash
from .summarize import build_summarizer, safe_batch
from .index import build_vectorstore, add_documents

def ingest_folder():
    cfg = load_settings()
    raw_dir = cfg["paths"]["raw_dir"]
    staged_dir = cfg["paths"]["staged_dir"]
    os.makedirs(staged_dir, exist_ok=True)

    retriever, vs = build_vectorstore(cfg)

    summarizer = build_summarizer(cfg)   # opcional

    # simples: ingere PDFs (você pode expandir p/ .png/.jpg usando outro partition)
    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(raw_dir, fname)

        print(">> Extraindo:", fpath)
        extracted = extract_from_pdf(fpath, cfg)

        # sumarizar só textos longos para reduzir custo
        texts = extracted["texts"]
        summaries = None
        if texts:
            print("   sum texts:", len(texts))
            sum_texts = safe_batch(summarizer, texts, batch_size=1)
            summaries = {"texts": sum_texts}

        # grava no staged (debug/reprodutibilidade)
        meta = {
            "file": fpath,
            "hash": file_hash(fpath),
            "counts": {k: len(v) if isinstance(v, list) else 0 for k,v in extracted.items()},
        }
        with open(os.path.join(staged_dir, fname + ".json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "extracted": extracted, "summaries": summaries}, f, ensure_ascii=False, indent=2)

        print("   indexando…")
        add_documents(retriever, vs, extracted, fpath, summaries=summaries)
        print("   OK")

    print("Pronto. Base atualizada.")
