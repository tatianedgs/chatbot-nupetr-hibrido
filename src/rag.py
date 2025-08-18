# -*- coding: utf-8 -*-
import os
from io import BytesIO
from typing import List, Tuple, Optional

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def chunk_pdf(pdf_bytes: bytes, filename: str) -> Tuple[List[str], List[dict]]:
    try:
        tipo_licenca, tipo_empreendimento = os.path.splitext(os.path.basename(filename))[0].split("_")
    except ValueError:
        tipo_licenca, tipo_empreendimento = "", ""

    reader = PdfReader(BytesIO(pdf_bytes))

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=700,
        chunk_overlap=120,
        length_function=len,
    )

    chunks, metas = [], []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        if not raw.strip():
            continue
        for piece in splitter.split_text(raw):
            chunks.append(piece)
            metas.append({
                "filename": filename,
                "tipo_licenca": tipo_licenca,
                "tipo_empreendimento": tipo_empreendimento,
                "page": i,
            })
    return chunks, metas


def build_or_update_index(kb, chunks: List[str], metas: List[dict], embeddings):
    if kb is None:
        kb = FAISS.from_texts(chunks, embeddings, metadatas=metas)
    else:
        kb.add_texts(chunks, metadatas=metas)
    return kb


def retrieve(kb, question: str, tipo_licenca: Optional[str] = None, tipo_empreendimento: Optional[str] = None):
    docs_scores = kb.similarity_search_with_score(question, k=12)

    filtered = []
    for d, s in docs_scores:
        meta = d.metadata or {}
        if tipo_licenca and meta.get("tipo_licenca") != tipo_licenca:
            continue
        if tipo_empreendimento and meta.get("tipo_empreendimento") != tipo_empreendimento:
            continue
        filtered.append((d, s))

    if not filtered:
        return [], None

    filtered.sort(key=lambda x: x[1])  # FAISS: menor distância = mais próximo

    top = filtered[:6]
    from collections import Counter
    pages = [d.metadata.get("page", 0) for d, _ in top]
    page_num = Counter(pages).most_common(1)[0][0]

    context_docs = [d for d, _ in top if d.metadata.get("page", -1) == page_num] or [d for d, _ in top[:3]]
    contexto = "\n\n---\n".join([c.page_content.strip() for c in context_docs])
    return [Document(page_content=contexto)], page_num
