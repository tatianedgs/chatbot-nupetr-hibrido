# -*- coding: utf-8 -*-
"""
NUPETR/IDEMA — Chat de Parecer Técnico (RAG)
- FAISS em memória (sem persistência)
- Botões: limpar histórico + exportar conversa (PDF via fpdf2)
- Sidebar: instruções para obter chave OpenAI e links oficiais
- Tema IDEMA (tons de verde) + logo centralizado + bolhas de chat
"""

# Evita conflito OpenMP em Windows/CPU
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
from io import BytesIO
from html import escape

import streamlit as st
from PIL import UnidentifiedImageError
from fpdf import FPDF

# Módulos próprios (RAG + LLM router)
from src.rag import chunk_pdf, build_or_update_index, retrieve
from src.llm_router import make_embeddings, make_chain_openai, LiteLocal


# -----------------------------
# Página / Tema (tons de verde)
# -----------------------------
st.set_page_config(
    page_title="NUPETR/IDEMA — Chat de Parecer Técnico",
    page_icon="🛢️",
    layout="wide"
)

# CSS do app (cores, banner, bolhas)
st.markdown("""
<style>
:root{
  --verde1:#2E7D32; /* primário */
  --verde2:#66BB6A; /* gradiente */
  --verde-bg:#F4F9F4; /* fundo */
  --verde-sec:#E8F3E8; /* cards */
  --texto:#0F2310;
}
[data-testid="stAppViewContainer"]{
  background: var(--verde-bg);
}
.nupetr-header{
  padding:16px 20px; margin:10px 0 14px 0; border-radius:14px;
  color:#fff; font-weight:700; font-size:20px;
  background: linear-gradient(90deg,var(--verde1),var(--verde2));
  display:flex; align-items:center; justify-content:center; gap:10px;
}

/* Bolhas do chat */
.user-bubble {
  background: #E6F4EA;  /* verde bem claro */
  border-radius: 12px;
  padding: 10px 12px;
}
.assistant-bubble {
  background: #FFFFFF;
  border-radius: 12px;
  padding: 10px 12px;
  border: 1px solid #DDE6DD;
}
.chat-gap { margin-bottom: 8px; }

/* Caixa lateral de ajuda */
.help-box{
  background: var(--verde-sec);
  padding: 10px 12px; border-radius: 10px; font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------
# Estado da sessão
# -----------------
if 'kb' not in st.session_state: st.session_state.kb = None
if 'pdfs_processed' not in st.session_state: st.session_state.pdfs_processed = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'mode' not in st.session_state: st.session_state.mode = 'leve'     # 'openai' | 'leve'
if 'openai_key' not in st.session_state: st.session_state.openai_key = ''
if 'openai_model' not in st.session_state: st.session_state.openai_model = 'gpt-4o-mini'
if 'lite_model' not in st.session_state: st.session_state.lite_model = None


# ---------------------------------
# Utilidades: limpar e exportar PDF
# ---------------------------------
def clear_history():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Como posso ajudar com suas dúvidas sobre pareceres?"
    }]

def _sanitize_for_pdf(text: str) -> str:
    """
    fpdf2 usa fontes core (Latin-1). Para evitar caracteres quebrados,
    convertemos para latin-1 com replacement.
    """
    return text.encode("latin-1", "replace").decode("latin-1")

def export_chat_to_pdf(messages) -> bytes:
    """
    Gera PDF simples com pares Você/Assistente.
    Cabeçalho com logo centralizado se existir.
    """
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Logo centralizado (opcional)
    logo_path = "img/idema.jpeg"
    if os.path.exists(logo_path):
        try:
            # Largura página ~210mm. Logo ~30mm de largura.
            pdf.image(logo_path, x=(210-30)/2, y=12, w=30)
            pdf.ln(30)  # espaço após logo
        except Exception:
            pdf.ln(10)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _sanitize_for_pdf("NUPETR/IDEMA — Conversa do Chat"), ln=1, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(2)

    for m in messages:
        who = "Você" if m["role"] == "user" else "Assistente"
        txt = f"{who}: {m['content']}"
        pdf.multi_cell(0, 6, _sanitize_for_pdf(txt))
        pdf.ln(2)

    # Retorna bytes
    out = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    out.write(pdf_bytes)
    return out.getvalue()


# ---------------
# Barra lateral
# ---------------
with st.sidebar:
    # Logo (com fallback para evitar erro em imagem corrompida)
    logo_path = "img/idema.jpeg"
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, width=160, caption="IDEMA/RN")
        except UnidentifiedImageError:
            st.warning("Logo inválido em img/idema.jpeg — seguindo sem imagem.")
            st.markdown("### IDEMA/RN")
    else:
        st.markdown("### IDEMA/RN")

    st.title("Pareceres Técnicos – NUPETR")
    st.caption("Assistente RAG para PDFs internos")

    # Ajuda rápida (caixa fixa)
    st.markdown("""
<div class="help-box">
<b>Como usar</b><br/>
1) Faça upload dos PDFs (padrão <i>tipoLicenca_tipoEmpreendimento.pdf</i>).<br/>
2) Escolha o modelo (OpenAI com chave, ou Leve sem chave).<br/>
3) Clique em <b>Processar PDFs</b> e depois pergunte no chat.
</div>
""", unsafe_allow_html=True)

    # Upload de PDFs
    st.subheader("📄 PDFs")
    pdfs = st.file_uploader(
        "Selecione (padrão: tipoLicenca_tipoEmpreendimento.pdf)",
        type=["pdf"], accept_multiple_files=True
    )

    # Escolha do modo
    st.subheader("🛠️ Modo do modelo")
    modo = st.radio("Escolha o modo:", ["OpenAI (com chave)", "Modelo Leve (sem chave)"])
    if modo.startswith("OpenAI"):
        st.session_state.mode = 'openai'
        st.session_state.openai_key = st.text_input(
            "OPENAI_API_KEY", type="password", help="Cole a chave (começa com sk-)"
        )
        # Seleção do modelo (nomes amigáveis aceitos; llm_router faz alias)
        st.session_state.openai_model = st.selectbox(
            "Modelo OpenAI",
            ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4.1"],
            index=0,
            help="gpt-5* serão mapeados para um modelo estável automaticamente."
        )
        with st.expander("🔑 Como obter sua OpenAI API Key", expanded=False):
            st.markdown(
                "- Acesse **https://platform.openai.com**\n"
                "- No menu, **API Keys** → **Create new secret key**\n"
                "- Copie a chave (formato `sk-...`) e cole no campo acima\n"
                "- No Streamlit Cloud, salve em **Settings → Secrets** para segurança."
            )
            st.link_button("Abrir página de API Keys", "https://platform.openai.com/account/api-keys")
            st.link_button("Guia rápido (oficial)", "https://platform.openai.com/docs/quickstart/create-and-export-an-api-key")
    else:
        st.session_state.mode = 'leve'
        st.session_state.openai_key = ''

    # Processar PDFs
    if st.button("🔄 Processar PDFs") and pdfs:
        with st.spinner("Processando PDFs..."):
            embeddings = make_embeddings(st.session_state.mode, st.session_state.openai_key)
            all_chunks, all_metas = [], []
            for pdf in pdfs:
                b = pdf.read(); pdf.seek(0)
                chunks, metas = chunk_pdf(b, pdf.name)
                all_chunks += chunks; all_metas += metas
            if all_chunks:
                st.session_state.kb = build_or_update_index(
                    st.session_state.kb, all_chunks, all_metas, embeddings
                )
                st.session_state.pdfs_processed = True
                st.success("PDFs processados. Preencha os filtros e pergunte no chat.")
            else:
                st.warning("Nenhum texto legível extraído.")

    # Ações do chat (limpar/exportar)
    st.subheader("💬 Ações do chat")
    st.button("🧹 Limpar histórico", on_click=clear_history)
    if st.button("📥 Exportar conversa (PDF)") and st.session_state.messages:
        pdf_bytes = export_chat_to_pdf(st.session_state.messages)
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64}" download="chat_nupetr.pdf">Clique aqui para baixar o PDF</a>',
            unsafe_allow_html=True
        )


# -------------
# Cabeçalho topo
# -------------
# Logo centralizado + banner
colL, colC, colR = st.columns([1,2,1])
with colC:
    # Tenta exibir a logo central; se não existir, só o banner
    logo_path = "img/idema.jpeg"
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, width=120)
        except UnidentifiedImageError:
            pass
st.markdown('<div class="nupetr-header">🛢️ NUPETR/IDEMA — Chat de Parecer Técnico</div>', unsafe_allow_html=True)
st.caption("As respostas citam trechos do PDF. Valide sempre as informações.")

# --------
# Filtros
# --------
c1, c2 = st.columns(2)
with c1:
    tipo_licenca = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
with c2:
    tipo_empreendimento = st.text_input("Tipo de Empreendimento", placeholder="ex.: POCO")

# -------------------------
# Histórico + Chat principal
# -------------------------
if not st.session_state.messages:
    clear_history()

# Render histórico com bolhas
for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="user-bubble chat-gap">{escape(m["content"])}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🛢️"):
            st.markdown(f'<div class="assistant-bubble chat-gap">{escape(m["content"])}</div>', unsafe_allow_html=True)

# Campo de entrada
user_q = st.chat_input("Digite sua pergunta aqui…")

# Tratamento da nova pergunta
if user_q:
    # Mostra a pergunta do usuário
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-bubble chat-gap">{escape(user_q)}</div>', unsafe_allow_html=True)
    # Guarda no histórico
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Verifica base vetorial
    if st.session_state.kb is None:
        st.warning("⚠️ Primeiro, carregue e processe PDFs na barra lateral.")
    else:
        # Recupera contexto (página dominante)
        docs, page_num = retrieve(
            st.session_state.kb,
            user_q,
            tipo_licenca or None,
            tipo_empreendimento or None
        )

        if not docs:
            answer = "Não encontrei essa informação nos PDFs carregados."
        else:
            contexto = docs[0].page_content
            if st.session_state.mode == 'openai' and st.session_state.openai_key:
                try:
                    chain = make_chain_openai(
                        st.session_state.openai_key,
                        model=st.session_state.openai_model,
                        temperature=0
                    )
                    answer = chain.run(input_documents=docs, question=user_q).strip()
                except Exception:
                    answer = "Tive um problema ao chamar a OpenAI. Verifique a chave e a internet e tente novamente."
            else:
                # Modo leve (sem chave)
                try:
                    if st.session_state.lite_model is None:
                        with st.spinner("Carregando modelo leve... (primeira vez pode demorar)"):
                            # sem parâmetro -> usa distilgpt2 (compatível no Streamlit Cloud)
                            st.session_state.lite_model = LiteLocal()
                    answer = st.session_state.lite_model.answer(contexto, user_q)
                except Exception as e:
                    answer = "Não consegui usar o modelo leve nesta sessão."

        # Mostra a resposta com a bolha
        with st.chat_message("assistant", avatar="🛢️"):
            st.markdown(f'<div class="assistant-bubble chat-gap">{escape(answer)}</div>', unsafe_allow_html=True)

        # Guarda no histórico
        st.session_state.messages.append({"role": "assistant", "content": answer})
