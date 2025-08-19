from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
import time
from dotenv import load_dotenv
import os

load_dotenv()

def build_summarizer():
    model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
        model="llama-3.1-8b-instant",
    )
    
    prompt_text = """
    You are an assistant. If the input is a question, answer it clearly and concisely.
    If the input is a piece of text or table, summarize it in portuguese.

    Input:
    {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    print("Deu bom!")
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def safe_batch(data, batch_size=1, wait=8):
    
    model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
        model="llama-3.1-8b-instant",
    )
    
    prompt_text = """
    You are an assistant. If the input is a question, answer it clearly and concisely.
    If the input is a piece of text or table, summarize it in portuguese.

    Input:
    {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    
    out = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        try:
            out.extend(summarize_chain.batch(batch, {"max_concurrency": 2}))
        except Exception as e:
            print("Rate/erro:", e, "retry…")
            time.sleep(wait)
            try:
                out.extend(summarize_chain.batch(batch, {"max_concurrency": 1}))
            except Exception as e2:
                print("falhou:", e2)
    return out


def safe_batch_process(data, batch_size=30, wait_on_error=10):
    model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
        model="llama-3.1-8b-instant",
    )
    
    prompt_text = """
    You are an assistant. If the input is a question, answer it clearly and concisely.
    If the input is a piece of text or table, summarize it in portuguese.

    Input:
    {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        try:
            batch_results = summarize_chain.batch(batch, {"max_concurrency": 2})
            results.extend(batch_results)
        except Exception as e:
            print(f"⏱️ Rate limit atingido ou erro: {e}")
            print(f"🔁 Esperando {wait_on_error} segundos para tentar de novo...")
            time.sleep(wait_on_error)
            # tenta o mesmo batch de novo
            try:
                batch_results = summarize_chain.batch(batch, {"max_concurrency": 2})
                results.extend(batch_results)
            except Exception as e2:
                print(f"❌ Ainda deu erro: {e2} — pulando esse batch.")
    return results