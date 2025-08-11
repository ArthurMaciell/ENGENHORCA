from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
import time

def build_summarizer(cfg):
    model = ChatGroq(
        api_key=cfg["env"]["GROQ_API_KEY"],
        temperature=0.2,
        model=cfg["llm"]["model"],
    )
    
    
    prompt_text = """
    You are an assistant. If the input is a question, answer it clearly and concisely.
    If the input is a piece of text or table, summarize it.

    Input:
    {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def safe_batch(chain: Runnable, data, batch_size=1, wait=8):
    out = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        try:
            out.extend(chain.batch(batch, {"max_concurrency": 2}))
        except Exception as e:
            print("Rate/erro:", e, "retry…")
            time.sleep(wait)
            try:
                out.extend(chain.batch(batch, {"max_concurrency": 1}))
            except Exception as e2:
                print("falhou:", e2)
    return out

