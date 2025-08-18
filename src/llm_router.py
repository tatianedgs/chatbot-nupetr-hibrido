# -*- coding: utf-8 -*-
"""
Roteador de LLMs e Embeddings para o Chat NUPETR/IDEMA.

- Embeddings:
    * OpenAI (text-embedding-3-small), quando modo='openai' e chave presente
    * Fallback sem chave: sentence-transformers/all-MiniLM-L6-v2

- LLM (QA):
    * OpenAI (ChatOpenAI via langchain-openai) -> load_qa_chain com QA_PROMPT
    * LiteLocal (sem chave, CPU): modelo pequeno via transformers (padrão: distilgpt2)
      -> evita 'sentencepiece' e compilações nativas
      -> se receber "google/mt5-small" ou "t5", faz fallback automático pra distilgpt2

Observações:
- Este módulo não depende de reportlab nem de libs nativas.
- Se você quiser um modelo PT melhor sem chave e aceitar download maior:
  * troque 'distilgpt2' por 'pierreguillou/gpt2-small-portuguese' no __init__ do LiteLocal.
"""

from typing import Optional, Dict
import os

# Embeddings e Chat OpenAI (SDK moderno do LangChain)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Modelo leve local (sem chave)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Prompt de QA (seu template)
from .ui_texts import QA_PROMPT


# =========================
# Util: alias para modelos
# =========================

# Alguns usuários escrevem "gpt-5", "gpt-5-mini"... caso não exista no momento,
# mapeamos para um modelo estável para evitar erro de API.
_OPENAI_MODEL_ALIAS: Dict[str, str] = {
    "gpt-5": "gpt-4o-mini",
    "gpt-5-mini": "gpt-4o-mini",
    "gpt-5-nano": "gpt-4o-mini",
}


def _resolve_openai_model(name: Optional[str]) -> str:
    """Resolve nomes amigáveis para modelos suportados."""
    if not name:
        return "gpt-4o-mini"
    key = name.strip().lower()
    return _OPENAI_MODEL_ALIAS.get(key, name)


# =====================
# Embeddings
# =====================

def make_embeddings(mode: str, openai_api_key: Optional[str]):
    """
    Retorna a função de embeddings de acordo com o modo.

    mode: 'openai' | 'leve' (qualquer outro cai no fallback local)
    """
    if mode == "openai" and openai_api_key:
        return OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    # Fallback sem chave (leve e confiável no Cloud)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# =====================
# LLMs (QA)
# =====================

def make_chain_openai(openai_api_key: str, model: Optional[str] = None, temperature: float = 0.0):
    """
    Cria uma cadeia de QA usando modelos da OpenAI via langchain-openai.
    - 'model' aceita nomes como 'gpt-4o-mini'. Se receber 'gpt-5*', será
      resolvido para um modelo estável via _resolve_openai_model.
    """
    resolved = _resolve_openai_model(model)
    llm = ChatOpenAI(api_key=openai_api_key, model=resolved, temperature=temperature)
    return load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)


class LiteLocal:
    """
    Modelo leve sem chave (Transformers) – compatível com Streamlit Cloud (CPU).

    Padrão: 'distilgpt2' (pequeno e rápido). Gera em português razoavelmente com prompt certo.
    Se quiser melhor PT (maior download), você pode usar:
      'pierreguillou/gpt2-small-portuguese'

    Fallback automático:
    - Se alguém passar 'google/mt5-small' ou 't5', trocamos para 'distilgpt2' para evitar
      dependências de 'sentencepiece' (que complica no Cloud).
    """
    def __init__(self, model_name: Optional[str] = None):
        # Escolha do modelo (env > argumento > padrão)
        env_name = os.getenv("NUPETR_LITE_MODEL", "").strip()
        name = (model_name or env_name or "distilgpt2").strip()

        if "t5" in name.lower():  # evita sentencepiece/compilações
            name = os.getenv("NUPETR_LITE_SAFE_MODEL", "distilgpt2")

        self.model_name = name
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.mdl = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Ajuste para evitar warnings de pad_token
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # CPU por padrão (Cloud)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.to(self.device)
        self.mdl.eval()

    def answer(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 180,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Gera uma resposta curta usando apenas o contexto (estilo Q/A).
        Para tornar mais determinístico no Cloud, 'do_sample'=False por padrão.
        """
        prompt = (
            "Você é um assistente técnico do IDEMA. Responda em português, curto e objetivo.\n"
            "Use apenas o CONTEXTO; se não houver informação suficiente, diga que não encontrou no documento.\n\n"
            f"=== CONTEXTO ===\n{context}\n=== FIM CONTEXTO ===\n\n"
            f"Pergunta: {question}\n"
            "Resposta:"
        )

        inputs = self.tok(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )

        with torch.no_grad():
            out = self.mdl.generate(**inputs, **gen_kwargs)

        text = self.tok.decode(out[0], skip_special_tokens=True)

        # heurstica simples: tenta recortar só o que vem após "Resposta:"
        if "Resposta:" in text:
            text = text.split("Resposta:", 1)[-1]

        # pequena limpeza
        return text.strip()
