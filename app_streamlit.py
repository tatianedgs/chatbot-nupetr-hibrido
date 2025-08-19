# -*- coding: utf-8 -*-
import os
import base64
from io import BytesIO

import streamlit as st
from PIL import UnidentifiedImageError

from src.rag import chunk_pdf, build_or_update_index, retrieve
from src.llm_router import make_embeddings, ask_openai, LiteLocal

# ---------- Página/tema ----------
st.set_page_config(page_title="NUPETR/IDEMA — Chat de Parecer Técnico", page_icon="🛢️", layout="wide")
st.markdown("""
<style>
.user-bubble{background:#E6F4EA;border-radius:12px;padding:10px 12px;}
.assistant-bubble{background:#FFFFFF;border:1px solid #DDE6DD;border-radius:12px;padding:10px 12px;}
.chat-gap{margin-bottom:8px;}
.nupetr-banner{padding:14px 18px;margin:6px 0 14px 0;border-radius:14px;color:#fff;font-weight:700;font-size:20px;
background:linear-gradient(90deg,#2E7D32,#66BB6A);display:flex;align-items:center;gap:10px;justify-content:center;}
</style>
""", unsafe_allow_html=True)

# ---------- Estado ----------
if 'kb' not in st.session_state: st.session_state.kb = None
if 'pdfs_processed' not in st.session_state: st.session_state.pdfs_processed = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'mode' not in st.session_state: st.session_state.mode = 'leve'   # openai | leve
if 'openai_key' not in st.session_state: st.session_state.openai_key = ''
if 'openai_model' not in st.session_state: st.session_state.openai_model = 'gpt-4o-mini'

# ---------- Utils: exportar PDF ----------
st.subheader("💬 Ações do chat")
st.button("🧹 Limpar histórico", on_click=clear_history)
if st.session_state.messages:
    html_bytes = export_chat_to_html(st.session_state.messages)
    st.download_button(
        "📥 Exportar conversa (HTML)",
        data=html_bytes,
        file_name="chat_nupetr.html",
        mime="text/html",
        help="Baixe e use 'Imprimir → Salvar como PDF' no navegador"
    )

def export_chat_to_html(messages) -> bytes:
    """Gera um HTML simples da conversa (pronto para 'Imprimir como PDF' no navegador)."""
    # tenta embutir o logo se existir
    logo_b64 = ""
    logo_path = "img/idema.jpeg"
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            logo_b64 = ""

    styles = """
    <style>
      body{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, 'Liberation Sans', sans-serif;
            background:#F4F9F4; color:#0F2310; margin:24px; }
      .banner{ padding:14px 18px; border-radius:14px; color:#fff; font-weight:700; font-size:20px;
               background:linear-gradient(90deg,#2E7D32,#66BB6A); display:flex; align-items:center; gap:10px; }
      .msg{ margin:10px 0; padding:10px 12px; border-radius:12px; }
      .user{ background:#E6F4EA; }
      .assistant{ background:#FFFFFF; border:1px solid #DDE6DD; }
      .who{ font-weight:600; margin-bottom:6px; }
    </style>
    """

    logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" width="100" style="vertical-align:middle;border-radius:8px" />' if logo_b64 else ""
    header = f"""
    <div class="banner">{logo_html}<span>🛢️ NUPETR/IDEMA — Chat de Parecer Técnico</span></div>
    <p style="opacity:.8">As respostas citam trechos do PDF. Valide sempre as informações.</p>
    """

    body = []
    for m in messages:
        who = "Você" if m["role"] == "user" else "Assistente"
        klass = "user" if m["role"] == "user" else "assistant"
        # escapinho básico
        txt = (m["content"] or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<div class="msg {klass}"><div class="who">{who}</div><div>{txt}</div></div>')

    html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{header}{''.join(body)}</body></html>"
    return html.encode("utf-8")

def _on_page(canvas, doc):
    img_path = "img/idema.jpeg"
    try:
        from PIL import Image as PILImage
        pil = PILImage.open(img_path)
        w,h = pil.size
        tw = 1.2 * inch
        th = tw * h / w
        img = RLImage(img_path, width=tw, height=th)
        x = (doc.width - tw)/2 + doc.leftMargin
        img.drawOn(canvas, x, doc.height + 1 * inch)
    except Exception:
        pass

def export_chat_to_pdf(messages) -> bytes:
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
    pdf = buf.getvalue(); buf.close()
    return pdf

def clear_history():
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar com suas dúvidas sobre pareceres?"}]

# ---------- Sidebar ----------
with st.sidebar:
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

    st.subheader("📄 PDFs")
    pdfs = st.file_uploader(
        "Selecione (padrão: tipoLicenca_tipoEmpreendimento.pdf)",
        type=["pdf"], accept_multiple_files=True
    )

    st.subheader("🛠️ Modo do modelo")
    modo = st.radio("Escolha o modo:", ["OpenAI (com chave)", "Modelo Leve (sem chave)"])
    if modo.startswith("OpenAI"):
        st.session_state.mode = 'openai'
        st.session_state.openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Cole sua chave (sk-...)")
        st.session_state.openai_model = st.selectbox("Modelo", ["gpt-4o-mini","gpt-4.1","gpt-5","gpt-5-mini"], index=0)
        with st.expander("🔑 Como obter sua OpenAI API Key", expanded=False):
            st.markdown(
                "- Acesse a página **API Keys** da OpenAI e clique em **Create new secret key**.\n"
                "- Copie a chave (formato `sk-...`) e cole no campo acima.\n"
                "- Guarde sua chave em local seguro — não compartilhe."
            )
            st.link_button("Abrir API Keys", "https://platform.openai.com/account/api-keys")
    else:
        st.session_state.mode = 'leve'
        st.session_state.openai_key = ''

    if st.button("🔄 Processar PDFs") and pdfs:
        with st.spinner("Processando PDFs..."):
            # embeddings não são usados nesta versão, mas mantemos a chamada por compatibilidade
            _ = make_embeddings(st.session_state.mode, st.session_state.openai_key)
            all_chunks, all_metas = [], []
            for pdf in pdfs:
                data = pdf.read(); pdf.seek(0)
                chunks, metas = chunk_pdf(data, pdf.name)
                all_chunks += chunks; all_metas += metas
            if all_chunks:
                st.session_state.kb = build_or_update_index(st.session_state.kb, all_chunks, all_metas, _)
                st.session_state.pdfs_processed = True
                st.success("PDFs processados. Preencha os filtros e pergunte no chat.")
            else:
                st.warning("Nenhum texto legível extraído.")

    st.subheader("💬 Ações do chat")
    st.button("🧹 Limpar histórico", on_click=clear_history)
    if st.button("📥 Exportar conversa (PDF)") and st.session_state.messages:
        pdf_bytes = export_chat_to_pdf(st.session_state.messages)
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="chat_nupetr.pdf">Baixar conversa (PDF)</a>', unsafe_allow_html=True)

# ---------- Banner ----------
st.markdown('<div class="nupetr-banner">🛢️ NUPETR/IDEMA — Chat de Parecer Técnico</div>', unsafe_allow_html=True)
st.caption("As respostas citam trechos do PDF. Valide sempre as informações.")

# ---------- Filtros ----------
c1, c2 = st.columns(2)
with c1:
    tipo_licenca = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
with c2:
    tipo_empreendimento = st.text_input("Tipo de Empreendimento", placeholder="ex.: POCO")

# ---------- Histórico render ----------
if not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar com suas dúvidas sobre pareceres?"}]

for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="user-bubble chat-gap">{m["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-bubble chat-gap">{m["content"]}</div>', unsafe_allow_html=True)

# ---------- Chat ----------
user_q = st.chat_input("Digite sua pergunta aqui…")

if user_q:
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-bubble chat-gap">{user_q}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_q})

    if st.session_state.kb is None:
        st.warning("⚠️ Primeiro, carregue e processe PDFs no menu lateral.")
    else:
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
                    answer = ask_openai(
                        contexto, user_q,
                        api_key=st.session_state.openai_key,
                        model=st.session_state.openai_model
                    )
                except Exception:
                    answer = "Tive um problema ao chamar a OpenAI. Verifique a chave e a internet e tente novamente."
            else:
                # Modo leve extrativo
                if 'lite_model' not in st.session_state:
                    st.session_state.lite_model = LiteLocal(max_sentences=4)
                answer = st.session_state.lite_model.answer(contexto, user_q)

        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-bubble chat-gap">{answer}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})


