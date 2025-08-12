from io import BytesIO
import base64

import pytesseract
from PIL import Image as PILImage  # evita conflito de nomes
from unstructured.documents.elements import (
    Table,
    Image as USImage,
    CompositeElement,
)

def ocr_from_images_base64(images_b64):
    image_texts = []
    for b64 in images_b64:
        try:
            image_data = base64.b64decode(b64)
            image = PILImage.open(BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang="por+eng")
            image_texts.append(text)
        except Exception as e:
            print(f"❌ Erro ao processar imagem: {e}")
            image_texts.append(text.strip())
    return image_texts


def extract_elements(chunks):
    tables, texts, images_b64 = [], [], []

    for ch in chunks:
        # Tabela “pura”
        if isinstance(ch, Table):
            tables.append(ch)

        # Tabelas/Imagens aninhadas
        if hasattr(ch.metadata, "orig_elements") and ch.metadata.orig_elements:
            for el in ch.metadata.orig_elements:
                if isinstance(el, Table):
                    tables.append(el)
                # >>> AQUI: checar Image do unstructured, não PIL
                if isinstance(el, USImage) and getattr(el.metadata, "image_base64", None):
                    images_b64.append(el.metadata.image_base64)

        # Texto (CompositeElement)
        if isinstance(ch, CompositeElement):
            # troque para texts.append(ch) se você QUISER o objeto
            texts.append(getattr(ch, "text", ""))

        # (Opcional) imagem “pura” no nível do chunk
        if isinstance(ch, USImage) and getattr(ch.metadata, "image_base64", None):
            images_b64.append(ch.metadata.image_base64)
            
    print(f'O número de chunks é: {len(chunks)}')
    print(f'O número de tabelas é: {len(tables)}')
    print(f'O número de textos é: {len(texts)}')
    print(f'O número de imagens é: {len(images_b64)}')

    return tables, texts, images_b64

def tables_to_html(tables):
    html_list = []
    for t in tables:
        html = getattr(t.metadata, "text_as_html", None)
        if html:
            html_list.append(html)
    return html_list
