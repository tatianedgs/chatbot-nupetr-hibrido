# -*- coding: utf-8 -*-
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # evita conflito OpenMP no Windows

from typing import Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # novo pacote
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback
from langchain.chains.question_answering import load_qa_chain
from .ui_texts import QA_PROMPT

def make_embeddings(mode: str, openai_api_key: Optional[str]):
    if mode == "openai" and openai_api_key:
        return OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def make_chain_openai(openai_api_key: str):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)
    return load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)

class LiteLocal:
    def __init__(self, model_name: str = "google/mt5-small"):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def answer(self, context: str, question: str) -> str:
        from .ui_texts import QA_PROMPT
        prompt = QA_PROMPT.format(context=context, question=question)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.mdl.generate(**inputs, max_new_tokens=240)
        text = self.tok.decode(outputs[0], skip_special_tokens=True)
        return text.split("Resposta:", 1)[-1].strip() if "Resposta:" in text else text.strip()
