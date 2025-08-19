# -*- coding: utf-8 -*-
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .ui_texts import QA_PROMPT

__all__ = ["make_embeddings", "make_chain_openai", "LiteLocal"]

# =====================
# Embeddings
# =====================
def make_embeddings(mode: str, openai_api_key: Optional[str]):
    if mode == "openai" and openai_api_key:
        return OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    # Fallback sem chave
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# =====================
# LLMs (OpenAI + Local Leve)
# =====================
def make_chain_openai(openai_api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Cria uma QA chain (LangChain) usando ChatOpenAI.
    - model: você pode trocar para 'gpt-5' SE sua conta tiver acesso;
      caso contrário use 'gpt-4.1' ou 'gpt-4o-mini' (padrão).
    """
    llm = ChatOpenAI(api_key=openai_api_key, model=model, temperature=temperature)
    return load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)


class LiteLocal:
    """
    Modelo leve sem chave (Transformers) – roda em CPU.
    Bom para o Streamlit Cloud quando não há API key.
    """
    def __init__(self, model_name: str = "google/mt5-small"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.to(self.device)

    def answer(self, context: str, question: str) -> str:
        prompt = QA_PROMPT.format(context=context, question=question)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.mdl.generate(**inputs, max_new_tokens=240)
        text = self.tok.decode(outputs[0], skip_special_tokens=True)
        return text.split("Resposta:", 1)[-1].strip() if "Resposta:" in text else text.strip()
