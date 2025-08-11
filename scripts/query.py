from src.rag import chain
from src.pipeline import ingest_folder
from src.pipeline import retriever  # ou de onde você carregou o retriever

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",)

if __name__ == "__main__":
    ingest_folder()
    rag_chain = chain(retriever)

    response = rag_chain.invoke(
        "Quais são os modelos de ventilador disponíveis?"
    )

    print(response)
