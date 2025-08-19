# app_streamlit.py
# -*- coding: utf-8 -*-
import os, base64
import streamlit as st
from PIL import UnidentifiedImageError

from src.rag import chunk_pdf, build_or_update_index, retrieve
from src.llm_router import make_embeddings, make_chain_openai, LiteLocal

st.set_page_config(page_title="NUPETR/IDEMA — Chat de Parecer Técnico", page_icon="🛢️", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
:root{--verde1:#2E7D32;--verde2:#66BB6A;--bg:#F4F9F4;--card:#E8F3E8;--txt:#0F2310;}
[data-testid="stAppViewContainer"]{background:var(--bg);}
.nupetr-banner{padding:14px 18px;margin:8px 0 12px 0;border-radius:14px;color:#fff;font-weight:700;font-size:20px;
background:linear-gradient(90deg,var(--verde1),var(--verde2));display:flex;align-items:center;gap:10px;justify-content:center;}
.user-bubble{background:#E6F4EA;border-radius:12px;padding:10px 12px;}
.assistant-bubble{background:#FFF;border:1px solid #DDE6DD;border-radius:12px;padding:10px 12px;}
.chat-gap{margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

# ---------- Estado ----------
if 'kb' not in st.session_state: st.session_state.kb = None
if 'pdfs_processed' not in st.session_state: st.session_state.pdfs_processed = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'mode' not in st.session_state: st.session_state.mode = 'leve'
if 'openai_key' not in st.session_state: st.session_state.openai_key = ''

# ---------- Utils: limpar/exportar (definidas ANTES) ----------
def clear_history():
    st.session_state.messages = [{"role":"assistant","content":"Como posso ajudar com suas dúvidas sobre pareceres?"}]

def export_chat_to_html(messages) -> bytes:
    logo_b64 = ""
    logo_path = "img/idema.jpeg"
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f: logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception: pass
    styles = """
    <style>
    body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,Noto Sans,'Liberation Sans',sans-serif;background:#F4F9F4;color:#0F2310;margin:24px;}
    .banner{padding:14px 18px;border-radius:14px;color:#fff;font-weight:700;font-size:20px;background:linear-gradient(90deg,#2E7D32,#66BB6A);display:flex;align-items:center;gap:10px;}
    .msg{margin:10px 0;padding:10px 12px;border-radius:12px}
    .user{background:#E6F4EA}.assistant{background:#FFF;border:1px solid #DDE6DD}
    .who{font-weight:600;margin-bottom:6px}
    </style>"""
    logo = f'<img src="data:image/jpeg;base64,{logo_b64}" width="100" style="vertical-align:middle;border-radius:8px" />' if logo_b64 else ""
    header = f'<div class="banner">{logo}<span style="margin-left:8px">🛢️ NUPETR/IDEMA — Chat de Parecer Técnico</span></div>'
    body = []
    for m in messages:
        who = "Você" if m["role"]=="user" else "Assistente"
        klass = "user" if m["role"]=="user" else "assistant"
        txt = (m["content"] or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        body.append(f'<div class="msg {klass}"><div class="who">{who}</div><div>{txt}</div></div>')
    html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{header}{''.join(body)}</body></html>"
    return html.encode("utf-8")

# ---------- Sidebar ----------
with st.sidebar:
    # logo com fallback
    logo_path = "img/idema.jpeg"
    if os.path.exists(logo_path):
        try: st.image(logo_path, width=160, caption="IDEMA/RN")
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
        st.session_state.openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Cole a chave (sk-...)")
        with st.expander("🔑 Como obter sua OpenAI API Key"):
            st.markdown(
                "- Abra **https://platform.openai.com**  \n"
                "- Menu **API Keys** → **Create new secret key**  \n"
                "- Copie a chave (`sk-...`) e cole acima.  \n"
                "- No Streamlit Cloud, salve em **Settings → Secrets**."
            )
    else:
        st.session_state.mode = 'leve'
        st.session_state.openai_key = ''

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

    st.subheader("💬 Ações do chat")
    st.button("🧹 Limpar histórico", on_click=clear_history)
    if st.session_state.messages:
        html_bytes = export_chat_to_html(st.session_state.messages)
        st.download_button("📥 Exportar conversa (HTML)", data=html_bytes,
                           file_name="chat_nupetr.html", mime="text/html",
                           help="Depois abra o HTML e use 'Imprimir → Salvar como PDF'.")

# ---------- Banner ----------
st.markdown('<div class="nupetr-banner">🛢️ NUPETR/IDEMA — Chat de Parecer Técnico</div>', unsafe_allow_html=True)
st.caption("As respostas citam trechos do PDF. Valide sempre as informações.")

# ---------- Filtros ----------
c1, c2 = st.columns(2)
with c1: tipo_licenca = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
with c2: tipo_empreendimento = st.text_input("Tipo de Empreendimento", placeholder="ex.: POCO")

# ---------- Histórico ----------
if not st.session_state.messages:
    clear_history()

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
        answer = "⚠️ Primeiro, carregue e processe PDFs no menu lateral."
    else:
        docs, page_num = retrieve(st.session_state.kb, user_q, tipo_licenca or None, tipo_empreendimento or None)
        if not docs:
            answer = "Não encontrei essa informação nos PDFs carregados."
        else:
            contexto = docs[0].page_content
            if st.session_state.mode == 'openai' and st.session_state.openai_key:
                try:
                    chain = make_chain_openai(st.session_state.openai_key)
                    answer = chain.run(input_documents=docs, question=user_q).strip()
                except Exception:
                    answer = "Tive um problema ao chamar a OpenAI. Verifique a chave e a internet."
            else:
                if 'lite_model' not in st.session_state:
                    with st.spinner("Carregando modelo leve... (pode demorar na primeira vez)"):
                        st.session_state.lite_model = LiteLocal("google/mt5-small")
                answer = st.session_state.lite_model.answer(contexto, user_q)

    with st.chat_message("assistant"):
        st.markdown(f'<div class="assistant-bubble chat-gap">{answer}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
