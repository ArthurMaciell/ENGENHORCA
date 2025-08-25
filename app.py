import streamlit as st
from dotenv import load_dotenv
#from scripts.helper import handler_user_input_image_rag,chunks_image,extract_elements,ocr_from_images_base64
from scripts.helper_simples import get_pdf_text, get_text_chunks,get_vectorstore,handler_user_input_simples
from scripts.helper_image import chunks_image, extract_elements,ocr_from_images_base64,get_conversation_chain_image_rag,handler_user_input_image_rag
#from scripts.summarize import build_summarizer,safe_batch,safe_batch_process
from scripts.index import build_vectorstore_from_extracted
#from scripts.rag import chain, chain_with_sources
from htmlTemplates import css,bot_template,user_template
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document

def main():
    load_dotenv()
    st.set_page_config(page_title='ENGENHORCA', page_icon='👨‍🔧')
    
    st.write(css, unsafe_allow_html=True)


    st.header('ENGENHORCA 👨‍🔧')
    tipo_leitura = st.selectbox('Você quer ler imagens e tabelas?',['Sim','Não'])
    if st.session_state.get("modo_atual") != tipo_leitura:
        st.session_state.modo_atual = tipo_leitura
        if tipo_leitura == "Não":
            st.session_state.conversation_text = None
            st.session_state.chat_history_text = []
        else:
            st.session_state.conversation_image = None
            st.session_state.chat_history_image = []
    
    user_question = st.text_input('Faça pergunta sobre o seu documento:')
    #if user_question:
        #if tipo_leitura == 'Não':
            #handler_user_input(user_question)
        #if tipo_leitura == 'Sim':
            #handler_user_input_image(user_question)
    
    with st.sidebar:
        st.subheader('Documentos')
        st.session_state.pdf_docs = st.file_uploader('Baixe seu PDF aqui e clique em processar.', accept_multiple_files=True)
        
    if tipo_leitura == 'Não':
        if st.button('Ler PDFs'):
            with st.spinner('Processando'):
                #Pegando o texto do PDF
                raw_text = get_pdf_text(st.session_state.pdf_docs)
                
                #Pegando o chunks dos textos
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                #Criar o vectorstore
                st.session_state.retriever = get_vectorstore(text_chunks)
                
                
                
    if tipo_leitura == 'Sim':
        if st.button('Ler PDFs'):
            with st.spinner('Processando'):
                if st.session_state.pdf_docs:
                    # 1) CRIE O ÍNDICE UMA ÚNICA VEZ (fora do loop)
                    retrievers_docs = []

                    # (opcional) contadores p/ debug
                    total_chunks = total_tables = total_images = total_ocr = 0

                    for pdf in st.session_state.pdf_docs:
                        chunks = chunks_image(pdf)
                        tables, texts, images_b64 = extract_elements(chunks)
                        image_text = ocr_from_images_base64(images_b64)
                        extracted = {
                            "texts": texts,
                            "tables": tables,
                            "image_text": image_text,
                        }
                        total_tables = len(tables)
                        total_texts = len(texts)
                        total_images = len(images_b64)
                        total_ocr = len(image_text)
                        retriever_image = build_vectorstore_from_extracted(extracted, source_name=pdf.name)
                        st.session_state.retriever_image = retriever_image
                        

                    # 8) (opcional) resumo de extração
                    st.info(f"Acumulado • elementos: {total_texts} | tabelas: {total_tables} | imagens: {total_images} | OCR textos: {total_ocr}")
                        
                    
    if user_question:
        if tipo_leitura == 'Não':
            handler_user_input_simples(user_question,st.session_state.retriever)
        if tipo_leitura == 'Sim':
            handler_user_input_image_rag(user_question,st.session_state.retriever_image)
        
if __name__ == '__main__':
    main()