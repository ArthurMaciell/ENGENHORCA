import streamlit as st
from dotenv import load_dotenv
from scripts.helper import get_pdf_text, get_text_chunks,get_vectorstore,save_uploaded_file,handler_user_input_image, get_conversation_chain,handler_user_input,chunks_image,extract_elements,ocr_from_images_base64
from scripts.summarize import build_summarizer,safe_batch
from scripts.index import build_vectorstore, add_documents
from scripts.rag import chain, chain_with_sources
from htmlTemplates import css,bot_template,user_template
from unstructured.partition.pdf import partition_pdf

def main():
    load_dotenv()
    st.set_page_config(page_title='ENGENHORCA', page_icon='👨‍🔧')
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('ENGENHORCA 👨‍🔧')
    tipo_leitura = st.selectbox('Você quer ler imagens e tabelas?',['Sim','Não'])
    user_question = st.text_input('Faça pergunta sobre o seu documento:')
    if user_question:
        if tipo_leitura == 'Não':
            handler_user_input(user_question)
        if tipo_leitura == 'Sim':
            handler_user_input_image(user_question)
    
    with st.sidebar:
        st.subheader('Documentos')
        st.session_state.pdf_docs = st.file_uploader('Baixe seu PDF aqui e clique em processar.', accept_multiple_files=True)
        
    if tipo_leitura == 'Não':
        if st.button('Chunks'):
            with st.spinner('Processando'):
                #Pegando o texto do PDF
                raw_text = get_pdf_text(st.session_state.pdf_docs)
                
                #Pegando o chunks dos textos
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                #Criar o vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                #Criação de conversa chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
    if tipo_leitura == 'Sim':
        if st.button('Chunks'):
            with st.spinner('Processando chunks'):
                if st.session_state.pdf_docs:
                    for pdf in st.session_state.pdf_docs:
                        path = save_uploaded_file(pdf)
                        chunks = chunks_image(pdf)
                        print(len(chunks))
                        st.success(f"Chunks totais: {len(chunks)}")
                        tables, texts, images_b64 = extract_elements(chunks)
                        image_text = ocr_from_images_base64(images_b64)
                        print(f'O número de chunks é: {len(chunks)}')
                        print(f'O número de tabelas é: {len(tables)}')
                        print(f'O número de textos é: {len(texts)}')
                        print(f'O número de imagens é: {len(images_b64)}')
                        summarize_chain = build_summarizer()
                        text_summaries = safe_batch(summarize_chain,texts)
                        table_summaries = safe_batch(summarize_chain, tables)
                        image_summaries = safe_batch(summarize_chain, image_text)
                        
                        retriever, vs = build_vectorstore()
                        extracted = {
                        "texts": texts,                   # lista de strings (ou elem.text)       
                        "image_text": image_text,           # lista de strings vindas do OCR
                        "images": image_summaries,        # lista de resumos textuais das imagens (opcional)
                            }
                        summaries = {
                            "texts": [s for s in (text_summaries + table_summaries + image_summaries) if s and s.strip()]
                        }
                        add_documents(retriever, vs, extracted, path,summaries)
                        
                        #Criação de conversa chain
                        st.session_state.conversation = chain(retriever)                      
                        
                    
        
        
if __name__ == '__main__':
    main()