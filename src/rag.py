# -*- coding: utf-8 -*-
import os
from io import BytesIO
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimpleDoc:
    page_content: str
    metadata: Dict[str, Any]


class TfIdfKB:
    """Índice em memória usando TF-IDF (Cloud-safe)."""
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None           # sparse matrix [n_docs x n_terms]
        self.texts: List[str] = []   # chunks
        self.metas: List[dict] = []  # metadados por chunk

    def add_texts(self, chunks: List[str], metas: List[dict]):
        # Reajusta (refit) o vetorizador com TODOS os textos
        self.texts.extend(chunks)
        self.metas.extend(metas)
        self.vectorizer = TfidfVectorizer(
            max_features=35000,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def query(self, question: str, k: int = 12):
        if not self.texts:
            return []
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.matrix)[0]  # shape: (n_docs,)
        # pega top-k índices ordenados por similaridade
        idxs = sims.argsort()[::-1][:k]
        return [(self.texts[i], self.metas[i], float(sims[i])) for i in idxs]


def chunk_pdf(pdf_bytes: bytes, filename: str) -> Tuple[List[str], List[dict]]:
    """Extrai texto e quebra em chunks com metadados básicos."""
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        tipo_licenca, tipo_empreendimento = base.split("_", 1)
    except ValueError:
        tipo_licenca, tipo_empreendimento = "", ""

    reader = PdfReader(BytesIO(pdf_bytes))
    chunks, metas = [], []
    # chunking simples por página (com divisão em blocos ~700 chars)
    CHUNK_SIZE = 700
    OVERLAP = 120

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        raw = raw.strip()
        if not raw:
            continue

        start = 0
        while start < len(raw):
            end = min(len(raw), start + CHUNK_SIZE)
            piece = raw[start:end]
            if piece.strip():
                chunks.append(piece)
                metas.append({
                    "filename": filename,
                    "tipo_licenca": tipo_licenca,
                    "tipo_empreendimento": tipo_empreendimento,
                    "page": i + 1,  # 1-based
                })
            if end == len(raw):
                break
            start = end - OVERLAP  # sobreposição

    return chunks, metas


def build_or_update_index(kb: Optional[TfIdfKB], chunks: List[str], metas: List[dict], _embeddings_unused=None):
    """Cria/atualiza o índice TF-IDF em memória."""
    if kb is None:
        kb = TfIdfKB()
    if chunks:
        kb.add_texts(chunks, metas)
    return kb


def retrieve(
    kb: TfIdfKB,
    question: str,
    tipo_licenca: Optional[str] = None,
    tipo_empreendimento: Optional[str] = None,
):
    """
    Retorna um [SimpleDoc] com o contexto dominante (página mais citada)
    e o número da página principal para referência.
    """
    hits = kb.query(question, k=12)
    if not hits:
        return [], None

    # filtra por metadados se informado
    filtered = []
    for text, meta, score in hits:
        if tipo_licenca and meta.get("tipo_licenca") != tipo_licenca:
            continue
        if tipo_empreendimento and meta.get("tipo_empreendimento") != tipo_empreendimento:
            continue
        filtered.append((text, meta, score))

    if not filtered:
        return [], None

    # escolhe a página dominante entre os top
    from collections import Counter
    pages = [m["page"] for _, m, _ in filtered[:6]]
    page_num = Counter(pages).most_common(1)[0][0]

    # agrega os pedaços da página dominante (ou, se vazio, dos 3 melhores)
    page_texts = [t for t, m, _ in filtered if m["page"] == page_num]
    if not page_texts:
        page_texts = [t for t, _, _ in filtered[:3]]

    contexto = "\n\n---\n".join([s.strip() for s in page_texts if s.strip()])
    return [SimpleDoc(page_content=contexto, metadata={"page": page_num})], page_num
