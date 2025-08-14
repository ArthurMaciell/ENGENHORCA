import streamlit as st
from dotenv import load_dotenv
from scripts.helper import get_pdf_text, get_text_chunks,get_vectorstore, get_conversation_chain
from htmlTemplates import css,bot_template,user_template


def main():
    load_dotenv()
    st.set_page_config(page_title='ENGENHORCA', page_icon='👨‍🔧')
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('Engenhorca 👨‍🔧')
    st.text_input('Faça pergunta sobre o seu documento:')
    
    st.write(user_template.replace("{{MSG}}", 'Hello Engenhorca'), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", 'Hello Human'), unsafe_allow_html=True)
    
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
    
    st.session_state.conversation

if __name__ == '__main__':
    main()