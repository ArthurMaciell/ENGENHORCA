from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from base64 import b64decode
import os
from dotenv import load_dotenv
from src.helper import build_qa_chain
from store_index import retriever


app = Flask(__name__)

load_dotenv()

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2= os.environ.get('LANGCHAIN_TRACING_V2')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')


model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2
)

qa = build_qa_chain(retriever, model)
pergunta = "Anxiety"
print(qa.invoke(pergunta))


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg") or ""
    msg = msg.strip()
    if not msg:
        return jsonify({"error": "mensagem vazia"}), 400
    try:
        answer = qa.invoke(msg)  # << input é string
        # Se seu front espera texto puro:
        return answer
        # Ou, se preferir JSON:
        # return jsonify({"answer": answer})
    except Exception as e:
        # logue o erro se quiser
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)