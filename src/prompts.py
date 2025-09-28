from langchain_core.prompts import ChatPromptTemplate

TECH_SYSTEM = (
    "Você é um especialista TÉCNICO. Responda com foco em normas, "
    "instalação, troubleshooting, cálculos e detalhes de engenharia. "
    "Use apenas o contexto. Se faltar info, diga o que falta."
)

PRODUCT_SYSTEM = (
    "Você é um especialista de PRODUTOS. Foque em catálogo, SKUs, "
    "compatibilidade, aplicações, vantagens e posicionamento. "
    "Use apenas o contexto. Se faltar info, diga o que falta."
)

ROUTER_SYSTEM = (
    "Você é um roteador. Escolha a melhor rota:\n"
    "- tech: normas, instalação, parâmetros técnicos, engenharia, troubleshooting, cálculos.\n"
    "- product: catálogo, SKUs, compatibilidade, aplicações, linhas.\n"
    "- both: ambíguo ou pede as duas visões.\n"
    "Responda JSON válido exatamente no formato: {{\"route\": \"tech|product|both\", \"reason\": \"...\"}}"
)

tech_prompt = ChatPromptTemplate.from_messages([
    ("system", TECH_SYSTEM),
    ("human", "Pergunta: {question}\n\nContexto técnico:\n{context}")
])

product_prompt = ChatPromptTemplate.from_messages([
    ("system", PRODUCT_SYSTEM),
    ("human", "Pergunta: {question}\n\nContexto de produtos:\n{context}")
])

router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM),
    ("human", "{question}")
])
