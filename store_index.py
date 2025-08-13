from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
import uuid, os
from dotenv import load_dotenv

load_dotenv()

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2= os.environ.get('LANGCHAIN_TRACING_V2')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')


extracted_data = load_pdf(r"C:\Users\Orçamento\Desktop\ENGENHORCA\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

persist_dir = r"C:\Users\Orçamento\Desktop\ENGENHORCA\chroma"  # mude se quiser
os.makedirs(persist_dir, exist_ok=True)

vectorstore = Chroma(
    collection_name="hvac_v1",               # nome NOVO evita reusar config velha
    embedding_function=embeddings,
    persist_directory=persist_dir,
    client_settings=Settings(
        allow_reset=True,
        anonymized_telemetry=False,          # silencia telemetria
    ),
)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# text_chunks já é List[Document] (do seu text_split)
parents = []
children = []

for d in text_chunks:
    if not (d.page_content or "").strip():
        continue
    pid = str(uuid.uuid4())

    # Pai = o chunk original (ou o "documento completo" se você tiver)
    parents.append((pid, d))

    # Filho = o que vai pro vetor (pode ser o próprio chunk OU um resumo dele)
    child = Document(
        page_content=d.page_content,  # troque por "summary" se tiver
        metadata={**(d.metadata or {}), id_key: pid}
    )
    children.append(child)

# grava pais no docstore e filhos no vetor
retriever.docstore.mset(parents)
retriever.vectorstore.add_documents(children)


# 1) contar vetores na collection (funciona com o wrapper do LangChain)
print("Qtde vetores:", vectorstore._collection.count())

# 2) testar a busca (usa seu retriever):
print(retriever.get_relevant_documents("Quem é o autor?")[:1])