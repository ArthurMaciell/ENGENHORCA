# src/router.py
import os
from typing import Literal, Dict
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from .prompts import router_prompt

class RouteSchema(BaseModel):
    route: Literal["tech", "product", "both"] = Field(...)
    reason: str

def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não encontrado. Defina no .env ou em st.secrets.")
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)

def classify(question: str) -> Dict[str, str]:
    llm = _get_llm()
    classifier = (router_prompt | llm | JsonOutputParser(pydantic_object=RouteSchema)).with_config(
        tags=["ENGENHORCA", "router"],
        metadata={"component": "router", "target_schema": "RouteSchema"}
    )
    return classifier.invoke({"question": question})

def route(question: str, tech_chain, prod_chain) -> dict:
    decision = classify(question)  # {'route': 'tech', 'reason': '...'}
    r = decision["route"]
    if r == "tech":
        ans = tech_chain.invoke(question)
        return {"answer": ans, "sources": None, "route": r, "reason": decision["reason"]}
    if r == "product":
        ans = prod_chain.invoke(question)
        return {"answer": ans, "sources": None, "route": r, "reason": decision["reason"]}
    tech_ans = tech_chain.invoke(question)
    prod_ans = prod_chain.invoke(question)
    both = f"— TÉCNICO —\n{tech_ans}\n\n— PRODUTOS —\n{prod_ans}"
    return {"answer": both, "sources": None, "route": r, "reason": decision["reason"]}

