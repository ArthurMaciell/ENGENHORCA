import streamlit as st
from dotenv import load_dotenv
from scripts.helper import get_pdf_text, get_text_chunks,get_vectorstore, get_conversation_chain,handler_user_input
from htmlTemplates import css,bot_template,user_template


def main():
    load_dotenv()
    st.set_page_config(page_title='ENGENHORCA', page_icon='👨‍🔧')
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('ENGENHORCA 👨‍🔧')
    user_question = st.text_input('Faça pergunta sobre o seu documento:')
    if user_question:
        handler_user_input(user_question)
    

    
    with st.sidebar:
        st.subheader('Documentos')
        pdf_docs = st.file_uploader('Baixe seu PDF aqui e clique em processar.', accept_multiple_files=True)
        if st.button('Processando'):
            with st.spinner('Processando'):
                #Pegando o texto do PDF
                raw_text = get_pdf_text(pdf_docs)
                
                #Pegando o chunks dos textos
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                #Criar o vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                #Criação de conversa chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
    

if __name__ == '__main__':
    main()