import streamlit as st
from dotenv import load_dotenv
from scripts.helper import save_uploaded_file,handler_user_input_image, get_conversation_chain_simples, get_conversation_chain,chunks_image,extract_elements,ocr_from_images_base64
from scripts.helper_simples import get_pdf_text, get_text_chunks,get_vectorstore,get_conversation_chain_rag,handler_user_input_simples
from scripts.summarize import build_summarizer,safe_batch,safe_batch_process
from scripts.index import build_vectorstore, add_documents
from scripts.rag import chain, chain_with_sources
from htmlTemplates import css,bot_template,user_template
from unstructured.partition.pdf import partition_pdf

def main():
    load_dotenv()
    st.set_page_config(page_title='ENGENHORCA', page_icon='👨‍🔧')
    
    st.write(css, unsafe_allow_html=True)


    st.header('ENGENHORCA 👨‍🔧')
    tipo_leitura = st.selectbox('Você quer ler imagens e tabelas?',['Sim','Não'])
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
        if st.button('Chunks'):
            with st.spinner('Processando chunks'):
                if st.session_state.pdf_docs:
                    # 1) CRIE O ÍNDICE UMA ÚNICA VEZ (fora do loop)
                    retriever, vs = build_vectorstore()

                    # (opcional) contadores p/ debug
                    total_chunks = total_tables = total_images = total_ocr = 0

                    for pdf in st.session_state.pdf_docs:
                        path = save_uploaded_file(pdf)

                        # 2) Extrai elementos do PDF
                        elems = chunks_image(pdf)
                        st.success(f"Chunks totais (este PDF): {len(elems)}")
                        total_chunks += len(elems)

                        tables, texts, images_b64 = extract_elements(elems)

                        # 3) OCR de imagens
                        image_text = ocr_from_images_base64(images_b64) or []

                        # 4) DEBUG
                        print(f'O número de chunks é: {len(elems)}')
                        print(f'O número de tabelas é: {len(tables)}')
                        print(f'O número de textos é: {len(texts)}')
                        print(f'O número de imagens é: {len(images_b64)}')

                        total_tables += len(tables)
                        total_images += len(images_b64)
                        total_ocr    += len(image_text)
                        
                        text_summaries = safe_batch_process(texts)
                        #text_summaries
                        table_summaries = safe_batch_process(tables)
                        image_summaries = safe_batch_process(image_text)

                        
                        extracted = {
                            "texts": texts,       # lista de strings ou Elements de texto normal
                            "tables": tables,      # lista de strings ou Elements de tabelas
                            "image_text": image_text   # lista de strings do OCR das imagens
                        } 
                        summaries = {
                            "texts": text_summaries,
                            "tables": table_summaries,
                            "images": image_summaries
                        } 
                        add_documents(retriever, vs, extracted, path,summaries)
                    
                    
                    probe = "caixa terminal"
                    docs = retriever.get_relevant_documents(probe)
                    st.write("Probe docs recuperados:", len(docs))
                    if docs:
                        st.code(docs[0].page_content[:500])

                    # 7) Cria a chain UMA VEZ, já com tudo indexado
                    st.session_state.conversation = chain(retriever)

                    # 8) (opcional) resumo de extração
                    st.info(f"Acumulado • elementos: {total_chunks} | tabelas: {total_tables} | imagens: {total_images} | OCR textos: {total_ocr}| table_summaries: {len(table_summaries)}")
                        
                    
    if user_question:
        if tipo_leitura == 'Não':
            handler_user_input_simples(user_question,st.session_state.retriever)
        if tipo_leitura == 'Sim':
            handler_user_input_image(user_question)
        
if __name__ == '__main__':
    main()