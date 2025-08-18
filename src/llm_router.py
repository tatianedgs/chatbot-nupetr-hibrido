# -*- coding: utf-8 -*-
from typing import Optional
from dataclasses import dataclass
import re

# OpenAI oficial (1.x)
from openai import OpenAI

# para o modo leve extrativo (sem LLM)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# (mantido por compatibilidade – não usamos mais embeddings externos aqui)
def make_embeddings(mode: str, openai_api_key: Optional[str]):
    return None


def ask_openai(context: str, question: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """
    Faz uma resposta 'stuff' simples: contexto + pergunta.
    """
    client = OpenAI(api_key=api_key)
    prompt = (
        "Você é um assistente técnico do IDEMA. Responda SEMPRE em português, curto e objetivo, "
        "usando somente o CONTEXTO fornecido. Se não houver informação suficiente, diga que não encontrou no documento.\n\n"
        f"=== CONTEXTO ===\n{context}\n=== FIM CONTEXTO ===\n\n"
        f"Pergunta: {question}\n"
        "Resposta (máx. 6 linhas, cite página/arquivo se possível):"
    )
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text.strip()


@dataclass
class LiteLocal:
    """
    Modo leve SEM CHAVE: resposta extrativa.
    Seleciona as frases do contexto mais parecidas com a pergunta (TF-IDF cosseno).
    """
    max_sentences: int = 4

    def answer(self, context: str, question: str) -> str:
        # quebra o contexto em frases simples
        sents = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', context) if s.strip()]
        if not sents:
            return "Não encontrei essa informação nos PDFs carregados."
        # TF-IDF entre pergunta e frases
        vect = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), lowercase=True)
        X = vect.fit_transform(sents + [question])
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        # pega top N frases
        idxs = sims.argsort()[::-1][: self.max_sentences]
        top = [sents[i] for i in idxs if sents[i]]
        txt = " ".join(top).strip()
        return txt or "Não encontrei essa informação nos PDFs carregados."
