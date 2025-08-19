# src/llm_router.py
# -*- coding: utf-8 -*-
from typing import Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Prompt simples diretamente aqui para evitar import circular
QA_TEMPLATE = (
    "Você é um assistente técnico do IDEMA. Responda SEMPRE em português claro e objetivo.\n"
    "Use apenas o contexto abaixo; se não houver informação suficiente, diga que não encontrou no documento.\n\n"
    "=== CONTEXTO ===\n{context}\n=== FIM CONTEXTO ===\n\n"
    "Pergunta: {question}\n"
    "Resposta (máx. 6 linhas, cite a seção/página se possível):"
)

def make_embeddings(mode: str, openai_api_key: Optional[str]):
    """Seleciona embeddings: OpenAI (se houver chave) ou Sentence-Transformers."""
    if mode == "openai" and openai_api_key:
        return OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def make_chain_openai(openai_api_key: str):
    """Retorna uma QA chain para LangChain usando ChatOpenAI (gpt-4o-mini por padrão)."""
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(QA_TEMPLATE)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

class LiteLocal:
    """Modelo leve sem chave (Transformers) – bom para Streamlit Cloud (CPU)."""
    def __init__(self, model_name: str = "google/mt5-small"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.to(self.device)

    def answer(self, context: str, question: str) -> str:
        prompt = QA_TEMPLATE.format(context=context, question=question)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.mdl.generate(**inputs, max_new_tokens=240)
        text = self.tok.decode(outputs[0], skip_special_tokens=True)
        return text.split("Resposta:", 1)[-1].strip() if "Resposta:" in text else text.strip()
