# -*- coding: utf-8 -*-
"""
NUPETR/IDEMA — Chat de Parecer Técnico (RAG)
- Mantém FAISS em memória (sem persistência)
- Botões: limpar histórico + exportar conversa (PDF)
- Sidebar: instruções para obter chave OpenAI e usar Secrets no Streamlit Cloud
- Tema em tons de verde do IDEMA + banner centralizado
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # evita conflito OpenMP em Windows/CPU

import base64
from io import BytesIO

import streamlit as st
from PIL import UnidentifiedImageError

# RAG/LLM (seu código local)
from src.rag import chunk_pdf, build_or_update_index, retrieve
from src.llm_router import make_embeddings, make_chain_openai, LiteLocal

# Exportar PDF (conversa)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch


# -----------------------------
# Página e tema (tons de verde)
# -----------------------------
st.set_page_config(page_title="NUPETR/IDEMA — Chat de Parecer Técnico", page_icon="🛢️", layout="wide")
st.markdown("""
<style>
/* balões com leve distinção de fundo */
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

/* dá um respiro entre as mensagens */
.chat-gap { margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)



# -----------------
# Estado da sessão
# -----------------
if 'kb' not in st.session_state: st.session_state.kb = None
if 'pdfs_processed' not in st.session_state: st.session_state.pdfs_processed = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'mode' not in st.session_state: st.session_state.mode = 'leve'   # 'openai' | 'leve'
if 'openai_key' not in st.session_state: st.session_state.openai_key = ''


# ---------------------------------
# Utilidades: limpar e exportar PDF
# ---------------------------------
def clear_history():
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar com suas dúvidas sobre pareceres?"}]

def _on_page(canvas, doc):
    """Desenha o logo no topo de cada página do PDF exportado (se existir)."""
    img_path = "img/idema.jpeg"
    try:
        from PIL import Image as PILImage
        pil = PILImage.open(img_path)
        w,h = pil.size
        target_w = 1.2 * inch
        target_h = target_w * h / w
        img = RLImage(img_path, width=target_w, height=target_h)
        x = (doc.width - target_w)/2 + doc.leftMargin  # centraliza
        img.drawOn(canvas, x, doc.height + 1 * inch)
    except Exception:
        pass

def export_chat_to_pdf(messages) -> bytes:
    """Gera um PDF simples com pares Você/Assistente."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]; body.alignment = 4  # justify

    story = [Spacer(1, 0.5*inch)]
    for m in messages:
        who = "Você" if m["role"] == "user" else "Assistente"
        story.append(Paragraph(f"<b>{who}:</b> {m['content']}", body))
        story.append(Spacer(1, 0.15*inch))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    pdf = buf.getvalue()
    buf.close()
    return pdf


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
        st.session_state.openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Cole sua chave (começa com sk-)")
        # Ajuda rápida para obter a chave
        with st.expander("🔑 Como obter sua OpenAI API Key", expanded=False):
            st.markdown(
                "- Acesse a página **API Keys** da OpenAI e clique em **Create new secret key**.\n"
                "- Copie a chave (formato `sk-...`) e cole no campo acima.\n"
                "- Guarde sua chave em local seguro — não compartilhe.",
            )
            st.link_button("Abrir página de API Keys", "https://platform.openai.com/account/api-keys")
            st.caption("Dica: você também pode seguir o passo a passo do Quickstart da OpenAI.")
            st.link_button("Quickstart: criar/exportar a API key", "https://platform.openai.com/docs/quickstart/create-and-export-an-api-key")
            st.caption("Fontes oficiais: API Keys e Quickstart.")
        # # Como usar no Streamlit Cloud (secrets)
        # with st.expander("☁️ Usar a chave como Secret no Streamlit Cloud"):
        #     st.markdown(
        #         "No deploy via Streamlit Cloud, **não** commit sua chave no repositório.\n\n"
        #         "1. No painel do app → **Settings** → **Advanced settings** → **Secrets**\n"
        #         "2. Adicione:\n"
        #         "```toml\nOPENAI_API_KEY = \"sk-xxx\"\n```\n"
        #         "3. No código, você pode ler com `st.secrets[\"OPENAI_API_KEY\"]`."
        #     )
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
                st.session_state.kb = build_or_update_index(st.session_state.kb, all_chunks, all_metas, embeddings)
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
# Banner topo
# -------------
st.markdown('<div class="nupetr-banner">🛢️ NUPETR/IDEMA — Chat de Parecer Técnico</div>', unsafe_allow_html=True)
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
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar com suas dúvidas sobre pareceres?"}]

for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="user-bubble chat-gap">{m["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-bubble chat-gap">{m["content"]}</div>', unsafe_allow_html=True)
   


# Campo de entrada SEM usar ainda a variável
user_q = st.chat_input("Digite sua pergunta aqui…")

# Só usamos user_q se o usuário realmente enviou algo nesta rodada
if user_q:
    # Mostra a bolha do usuário imediatamente
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-bubble chat-gap">{user_q}</div>', unsafe_allow_html=True)

    # Guarda no histórico
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Verifica se já temos base vetorial
    if st.session_state.kb is None:
        st.warning("⚠️ Primeiro, carregue e processe PDFs no menu lateral.")
    else:
        # Recupera contexto
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
                    chain = make_chain_openai(st.session_state.openai_key)
                    answer = chain.run(input_documents=docs, question=user_q).strip()
                except Exception:
                    answer = "Tive um problema ao chamar a OpenAI. Verifique a chave e a internet e tente novamente."
            else:
                # Modo leve (sem chave)
                if 'lite_model' not in st.session_state:
                    with st.spinner("Carregando modelo leve... (pode demorar na primeira vez)"):
                        st.session_state.lite_model = LiteLocal("google/mt5-small")
                answer = st.session_state.lite_model.answer(contexto, user_q)

        # Mostra a resposta com a bolha/ícone
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-bubble chat-gap">{answer}</div>', unsafe_allow_html=True)

        # Guarda no histórico
        st.session_state.messages.append({"role": "assistant", "content": answer})

