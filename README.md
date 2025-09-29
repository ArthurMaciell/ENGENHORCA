# 🧭 ENGENHORCA — Multi-Agent RAG para HVAC

ENGENHORCA é um sistema de **agentes de IA** para auxiliar em dúvidas técnicas e comerciais sobre produtos HVAC.  
Ele combina **RAG (Retrieval-Augmented Generation)**, **roteamento por agentes** e **validação de dados** para oferecer respostas contextualizadas, com fontes e explicações.

---

## 🚀 Funcionalidades

- 🔀 **Router multi-agente**: decide entre agente técnico, agente de produtos ou ambos.  
- 📑 **Arquitetura RAG completa**: chunking de documentos, embeddings e recuperação em banco vetorial.  
- 📚 **Suporte a múltiplos tipos de documentos** (PDF, TXT, MD).  
- 🛠 **Pipeline em Python + LangChain**, pronto para evolução com **LangGraph**.  
- 🧾 **Validação de entrada e saída com Pydantic** (schemas `QueryRequest` e `QueryResponse`).  
- 📦 **Vector stores suportados**:
  - Local: FAISS / Chroma  
  - Cloud-ready: Pinecone / Weaviate (config em `vector_store_config.yaml`)  
- 🖥 **Interface em Streamlit** para interação amigável.  
- 🧩 **Modularidade**: retrievers, chains e router separados em módulos.

---

## 📂 Estrutura do projeto

```
ENGENHORCA/
│
├── app.py              # Interface principal (Streamlit)
├── src/
│   ├── chains.py       # Construção das chains (LangChain)
│   ├── retrievers.py   # Configuração dos retrievers vetoriais
│   ├── router.py       # Roteador (decide agente técnico/produto)
│   ├── ingest.py       # Ingestão e indexação de documentos
│   ├── prompts.py      # Prompts usados nos agentes
│   ├── schemas.py      # Pydantic models (entrada/saída)
│   └── settings.py     # Configurações via pydantic-settings
│
├── data/               # Documentos de entrada (PDF, txt, etc.)
├── vectorstores/       # Index persistido (Chroma/FAISS)
└── requirements.txt
```

---

## ⚙️ Como rodar

### 1. Clone e instale dependências
```bash
git clone https://github.com/seu-usuario/ENGENHORCA.git
cd ENGENHORCA
pip install -r requirements.txt
```

### 2. Configure variáveis de ambiente
Crie um `.env`:
```env
GROQ_API_KEY=coloque_sua_chave_aqui
```

### 3. Ingestão de documentos
Coloque arquivos em `data/tech/` e `data/products/`.  
Depois rode:
```bash
python -m src.ingest
```

### 4. Inicie o app
```bash
streamlit run app.py
```

---

## 🔍 Exemplo de uso

**Entrada**  
```
"Quais são as diferenças entre difusores lineares e grelhas fixas?"
```

**Saída**  
```json
{
  "answer": "Difusores lineares distribuem o ar em linha contínua, enquanto grelhas fixas direcionam em padrões definidos...",
  "sources": ["manual_tecnico.pdf", "catalogo_produtos.docx"],
  "route": "both",
  "reason": "A pergunta envolve aspectos técnicos e comerciais."
}
```

---

## 🛣 Roadmap

- [ ] Suporte a **LangGraph** para workflows multi-agente.  
- [ ] Integração com **LangSmith** para observabilidade.  
- [ ] Adaptador pronto para **Pinecone** e **Weaviate**.  
- [ ] Deploy via **FastAPI** ou **LangServe**.  
- [ ] CI/CD com GitHub Actions.  


---

## 📜 Licença
MIT License.  
