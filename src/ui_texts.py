# -*- coding: utf-8 -*-
from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você é um assistente técnico do IDEMA. Responda SEMPRE em português claro e objetivo.\n"
        "Use apenas o contexto abaixo; se não houver informação suficiente, diga que não encontrou no documento.\n\n"
        "=== CONTEXTO ===\n{context}\n=== FIM CONTEXTO ===\n\n"
        "Pergunta: {question}\n"
        "Resposta (máx. 6 linhas, cite a seção/página se possível):"
    ),
)
