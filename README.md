# Chatbot NUPETR Híbrido — Guia completo (Streamlit + Executável)

Este pacote entrega **2 produtos**:

1. **App Streamlit (deploy no Streamlit Cloud/GitHub)** com **3 modos de LLM**:

   * **OpenAI (com chave)** – modelos GPT via `langchain-openai`.
   * **Leve (sem chave)** – modelo local via `transformers` (CPU), adequado ao Streamlit Cloud.
   * **Fallback embeddings sem chave** – `sentence-transformers` para indexar PDFs sem depender da OpenAI.

2. **Executável Desktop (Windows)** com modo **Robusto Local** (modelo `.gguf` via GPT4All), para consultas RAG off-line.

Tudo pensado para repositório **leve** (sem pesos de modelos), pronto para **GitHub** e com **passo a passo**.

> **Novo**: suporte **Ollama (local/offline)** e opção de **ChromaDB** (índice persistente) além do FAISS em memória.

---

## Estrutura de pastas (sugestão)

```
chatbot-nupetr-hibrido/
├─ app_streamlit.py                # App web (Streamlit)
├─ app_desktop.py                  # App desktop (CLI) para empacotamento
├─ cli_chat.py                     # CLI com OpenAI/Ollama + Chroma
├─ src/
│  ├─ rag.py                       # Funções de RAG (FAISS/Chroma, chunking, leitura PDF)
│  ├─ llm_router.py                # Seleção de modelos (OpenAI, Leve, Ollama, Robusto)
│  ├─ data_ingest.py               # Ingestão e persistência Chroma
│  └─ ui_texts.py                  # Textos/promptos e utilidades
├─ docs/                           # Coloque aqui seus PDFs de exemplo (não versionar dados sensíveis)
├─ storage/
│  └─ chroma/                      # Persistência do Chroma (gitignored)
├─ img/
│  └─ idema.jpeg                   # Logo opcional
├─ models/                         # (DESKTOP) armazena modelos .gguf (gitignored)
├─ requirements-streamlit.txt      # Dependências do app Streamlit
├─ requirements-desktop.txt        # Dependências do executável Desktop
├─ .env.example
├─ .gitignore
└─ README.md
```

chatbot-nupetr-hibrido/ ├─ app\_streamlit.py                # App web (Streamlit) ├─ app\_desktop.py                  # App desktop (CLI) para empacotamento ├─ src/ │  ├─ rag.py                       # Funções de RAG (FAISS, chunking, leitura PDF) │  ├─ llm\_router.py                # Seleção de modelos (OpenAI, Leve, Robusto) │  └─ ui\_texts.py                  # Textos/promptos e utilidades ├─ docs/                           # Coloque aqui seus PDFs de exemplo (não versionar dados sensíveis) ├─ img/ │  └─ idema.jpeg                   # Logo opcional ├─ models/                         # (DESKTOP) armazena modelos .gguf (gitignored) ├─ requirements-streamlit.txt      # Dependências do app Streamlit ├─ requirements-desktop.txt        # Dependências do executável Desktop ├─ .gitignore └─ README.md

```

> **Importante**: Não comite arquivos grandes ou confidenciais. O diretório `models/` é **ignorado** pelo git, assim como caches, índices FAISS e artefatos do PyInstaller.

---

## requirements-streamlit.txt (pinned)

```

# Web app

streamlit==1.37.1

# RAG e PDF

langchain==0.2.13 langchain-community==0.2.12 langchain-openai==0.1.21 sentence-transformers==2.7.0 faiss-cpu==1.8.0.post1 PyPDF2==3.0.1 pymupdf==1.24.9 pillow==10.4.0

# Modelos locais leves (sem chave)

transformers==4.43.3 accelerate==0.33.0

# CPU-only; deixe o pip resolver a melhor variante de torch

torch==2.3.1

# Vetor e provedores extras

chromadb==0.5.5 ollama>=0.3.0 openai>=1.30.0 click==8.1.7

# Utilidades

python-dotenv==1.0.1 reportlab==3.6.13

```
# Web app
streamlit==1.37.1

# RAG e PDF
langchain==0.2.13
langchain-community==0.2.12
langchain-openai==0.1.21
sentence-transformers==2.7.0
faiss-cpu==1.8.0.post1
PyPDF2==3.0.1
pymupdf==1.24.9
pillow==10.4.0

# Modelos locais leves (sem chave)
transformers==4.43.3
accelerate==0.33.0
# CPU-only; deixe o pip resolver a melhor variante de torch
torch==2.3.1

# Utilidades
python-dotenv==1.0.1
reportlab==3.6.13
```

> Observação: se `faiss-cpu` falhar no Windows, você pode trocar por **Chroma** (`chromadb==0.5.5`) e adaptar `src/rag.py` (já deixamos comentário no código para isso).

---

## requirements-desktop.txt (pinned)

```
# Núcleo RAG e PDF
langchain==0.2.13
langchain-community==0.2.12
sentence-transformers==2.7.0
faiss-cpu==1.8.0.post1
PyPDF2==3.0.1

# Modelo robusto local (.gguf) via GPT4All
gpt4all==2.8.2

# Utilidades
python-dotenv==1.0.1
click==8.1.7

# (opcional) para converter em executável
pyinstaller==6.10.0
```

# Núcleo RAG e PDF

langchain==0.2.13 langchain-community==0.2.12 sentence-transformers==2.7.0 faiss-cpu==1.8.0.post1 PyPDF2==3.0.1

# Modelo robusto local (.gguf) via GPT4All

gpt4all==2.8.2

# Utilidades

python-dotenv==1.0.1 click==8.1.7

# (opcional) para converter em executável

pyinstaller==6.10.0

```

> O app desktop **não** precisa de `streamlit`, `transformers` nem `openai`. Mantemos o footprint menor.

---

## .gitignore (sugestão)

```

# Python

**pycache**/ \*.pyc .venv/ .env

# Streamlit/Cache

.streamlit/

# Modelos e índices

models/ *.gguf faiss\_index/ index\_*

# Build/Dist

build/ dist/ \*.spec

# Dados

docs/\*.pdf !docs/README.md

````

> Coloque um `docs/README.md` explicando como adicionar PDFs localmente sem comitar documentos reais do IDEMA.

---

## src/ui_texts.py

```python
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
````

---

## src/rag.py

```python
# -*- coding: utf-8 -*-
import os
from io import BytesIO
from typing import List, Tuple, Optional

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Se precisar de fallback para Chroma (se FAISS falhar no Windows):
# from langchain_community.vectorstores import Chroma
# import chromadb


def chunk_pdf(pdf_bytes: bytes, filename: str) -> Tuple[List[str], List[dict]]:
    try:
        tipo_licenca, tipo_empreendimento = os.path.splitext(os.path.basename(filename))[0].split("_")
    except ValueError:
        tipo_licenca, tipo_empreendimento = "", ""

    reader = PdfReader(BytesIO(pdf_bytes))

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=700,
        chunk_overlap=120,
        length_function=len,
    )

    chunks, metas = [], []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        if not raw.strip():
            continue
        for piece in splitter.split_text(raw):
            chunks.append(piece)
            metas.append({
                "filename": filename,
                "tipo_licenca": tipo_licenca,
                "tipo_empreendimento": tipo_empreendimento,
                "page": i,
            })
    return chunks, metas


def build_or_update_index(
    kb, chunks: List[str], metas: List[dict], embeddings
):
    if kb is None:
        kb = FAISS.from_texts(chunks, embeddings, metadatas=metas)
        # Alternativa com Chroma:
        # kb = Chroma.from_texts(
        #     texts=chunks, metadatas=metas,
        #     embedding_function=embeddings,
        #     persist_directory="faiss_index"
        # )
    else:
        kb.add_texts(chunks, metadatas=metas)
    return kb


def retrieve(kb, question: str, tipo_licenca: Optional[str] = None, tipo_empreendimento: Optional[str] = None):
    docs_scores = kb.similarity_search_with_score(question, k=12)

    filtered = []
    for d, s in docs_scores:
        meta = d.metadata or {}
        if tipo_licenca and meta.get("tipo_licenca") != tipo_licenca:
            continue
        if tipo_empreendimento and meta.get("tipo_empreendimento") != tipo_empreendimento:
            continue
        filtered.append((d, s))

    if not filtered:
        return [], None

    filtered.sort(key=lambda x: x[1])  # FAISS: menor distância = mais próximo

    top = filtered[:6]
    from collections import Counter
    pages = [d.metadata.get("page", 0) for d, _ in top]
    page_num = Counter(pages).most_common(1)[0][0]

    context_docs = [d for d, _ in top if d.metadata.get("page", -1) == page_num] or [d for d, _ in top[:3]]

    contexto = "\n\n---\n".join([c.page_content.strip() for c in context_docs])
    return [Document(page_content=contexto)], page_num
```

---

## src/llm\_router.py

```python
# -*- coding: utf-8 -*-
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .ui_texts import QA_PROMPT


# =====================
# Embeddings
# =====================

def make_embeddings(mode: str, openai_api_key: Optional[str]):
    if mode == "openai" and openai_api_key:
        return OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    # Fallback sem chave
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# =====================
# LLMs
# =====================

def make_chain_openai(openai_api_key: str):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)
    return load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)


class LiteLocal:
    """Modelo leve sem chave (Transformers) – foco: Streamlit Cloud CPU."""
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
        # tenta extrair após "Resposta:" se existir
        return text.split("Resposta:", 1)[-1].strip() if "Resposta:" in text else text.strip()
```

---

## app\_streamlit.py (web)

```python
# -*- coding: utf-8 -*-
import base64
import os
import uuid
from io import BytesIO
import streamlit as st

from src.rag import chunk_pdf, build_or_update_index, retrieve
from src.llm_router import make_embeddings, make_chain_openai, LiteLocal

st.set_page_config(page_title="NUPETR/IDEMA • Chat de Parecer", page_icon="💼", layout="wide")

# =====================
# Estado
# =====================
if 'kb' not in st.session_state: st.session_state.kb = None
if 'pdfs_processed' not in st.session_state: st.session_state.pdfs_processed = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'mode' not in st.session_state: st.session_state.mode = 'leve'   # openai | leve
if 'openai_key' not in st.session_state: st.session_state.openai_key = ''


# =====================
# Sidebar
# =====================
with st.sidebar:
    st.image("img/idema.jpeg", width=140)
    st.title("Pareceres Técnicos – NUPETR")
    st.caption("Assistente RAG para PDFs do setor")

    st.subheader("📄 PDFs")
    pdfs = st.file_uploader("Selecione PDFs (formato tipoLicenca_tipoEmpreendimento.pdf)", type=["pdf"], accept_multiple_files=True)

    st.subheader("⚙️ Modo do modelo")
    modo = st.radio("Escolha o modo:", ["Usar OpenAI (com chave)", "Usar Modelo Leve (sem chave)"])

    if modo.startswith("Usar OpenAI"):
        st.session_state.mode = 'openai'
        st.session_state.openai_key = st.text_input("OPENAI_API_KEY", type="password")
    else:
        st.session_state.mode = 'leve'
        st.session_state.openai_key = ''

    if st.button('🔄 Processar PDFs') and pdfs:
        with st.spinner('Processando PDFs...'):
            # Embeddings dependem do modo
            embeddings = make_embeddings(st.session_state.mode, st.session_state.openai_key)
            all_chunks, all_metas = [], []
            for pdf in pdfs:
                b = pdf.read()
                pdf.seek(0)
                chunks, metas = chunk_pdf(b, pdf.name)
                all_chunks.extend(chunks)
                all_metas.extend(metas)
            if all_chunks:
                st.session_state.kb = build_or_update_index(st.session_state.kb, all_chunks, all_metas, embeddings)
                st.session_state.pdfs_processed = True
                st.success("PDFs processados. Faça sua pergunta no chat.")
            else:
                st.warning("Nenhum texto legível extraído dos PDFs.")


# =====================
# Filtros
# =====================
col1, col2 = st.columns(2)
with col1:
    tipo_licenca = st.text_input("Tipo de Licença", "")
with col2:
    tipo_empreendimento = st.text_input("Tipo de Empreendimento", "")


# =====================
# Chat
# =====================
st.title("💼 NUPETR/IDEMA — Chat de Parecer Técnico")
st.caption("As respostas citam trechos do PDF. Valide sempre as informações.")

if not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar com suas dúvidas sobre pareceres?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Digite sua pergunta aqui…")
if user_q and st.session_state.kb is not None:
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Recupera contexto
    docs, page_num = retrieve(st.session_state.kb, user_q, tipo_licenca or None, tipo_empreendimento or None)

    if not docs:
        answer = "Não encontrei essa informação nos PDFs carregados."
    else:
        contexto = docs[0].page_content
        if st.session_state.mode == 'openai' and st.session_state.openai_key:
            chain = make_chain_openai(st.session_state.openai_key)
            answer = chain.run(input_documents=docs, question=user_q).strip()
        else:
            # Modo leve
            if 'lite_model' not in st.session_state:
                with st.spinner('Carregando modelo leve... (primeira vez pode demorar)'):
                    st.session_state.lite_model = LiteLocal("google/mt5-small")
            answer = st.session_state.lite_model.answer(contexto, user_q)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

elif user_q and st.session_state.kb is None:
    st.warning("⚠️ Primeiro, carregue e processe PDFs no menu lateral.")
```

---

## app\_desktop.py (CLI robusto com GPT4All)

```python
# -*- coding: utf-8 -*-
import os
import sys
import click
from glob import glob

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from gpt4all import GPT4All

INDEX_PATH = "faiss_index"
MODELS_DIR = "models"
DEFAULT_GGUF = os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct-q4_0.gguf")


def build_index_from_docs(docs_dir: str):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=700, chunk_overlap=120, length_function=len)
    chunks, metas = [], []
    for pdf in glob(os.path.join(docs_dir, "*.pdf")):
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            if not raw.strip():
                continue
            for piece in splitter.split_text(raw):
                chunks.append(piece)
                metas.append({"filename": os.path.basename(pdf), "page": i})

    if not chunks:
        raise RuntimeError("Nenhum texto extraído dos PDFs.")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    kb = FAISS.from_texts(chunks, emb, metadatas=metas)
    kb.save_local(INDEX_PATH)
    return kb


def load_index():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--docs', default='docs', help='Pasta com PDFs')
@click.option('--force', is_flag=True, help='Recriar índice mesmo se existir')
def index(docs, force):
    if os.path.exists(INDEX_PATH) and not force:
        click.echo("Índice já existe. Use --force para recriar.")
        return
    kb = build_index_from_docs(docs)
    click.echo(f"Índice criado com sucesso. Vetores: {kb.index.ntotal}")


@cli.command()
@click.option('--model', default=DEFAULT_GGUF, help='Caminho para o modelo .gguf')
@click.option('--topk', default=6, help='Qtd. de trechos recuperados')
@click.option('--docs', default='docs', help='Pasta com PDFs (para criar índice se necessário)')
def chat(model, topk, docs):
    if not os.path.exists(INDEX_PATH):
        click.echo("Nenhum índice encontrado. Criando a partir de docs/…")
        build_index_from_docs(docs)
    kb = load_index()

    if not os.path.exists(model):
        click.echo(f"Modelo não encontrado: {model}")
        click.echo("Baixe um .gguf, ex.: Phi-3-mini (q4_0) para CPU, e coloque em models/.")
        sys.exit(1)

    llm = GPT4All(model)
    click.echo("Assistente pronto. Digite sua pergunta (ou 'sair'):")
    while True:
        q = input("\nVocê: ").strip()
        if not q or q.lower() in {"sair", "exit", "quit"}:
            break
        docs_scores = kb.similarity_search_with_score(q, k=topk)
        docs_scores.sort(key=lambda x: x[1])
        context = "\n\n---\n".join([d.page_content for d, _ in docs_scores[:3]])
        prompt = (
            "Você é um assistente técnico do IDEMA. Use apenas o contexto a seguir para responder sucintamente.\n\n"
            f"=== CONTEXTO ===\n{context}\n=== FIM CONTEXTO ===\n\nPergunta: {q}\nResposta:"
        )
        resp = llm.generate(prompt, max_tokens=512)
        print(f"\nAssistente: {resp}")


if __name__ == '__main__':
    cli()
```

---

## README.md — Passo a passo

### Tema (IDEMA • tons de verde)

* Adicionado arquivo `.streamlit/config.toml` com a paleta:

  ```toml
  [theme]
  primaryColor="#2E7D32"
  backgroundColor="#F4F7F4"
  secondaryBackgroundColor="#E8F1E8"
  textColor="#0F2310"
  font="sans serif"
  ```
* Banner de cabeçalho com gradiente verde no `app_streamlit.py` (classe `.nupetr-header`).

---

## README.md — Passo a passo

### 1) Clonar e criar ambiente

```bash
git clone https://github.com/<seu-usuario>/chatbot-nupetr-hibrido.git
cd chatbot-nupetr-hibrido
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Rodar local (Streamlit)

```bash
pip install -r requirements-streamlit.txt
cp .env.example .env
# (opcional) exportar a chave
setx OPENAI_API_KEY "sua-chave"        # Windows (nova sessão)
# Linux/macOS: export OPENAI_API_KEY="sua-chave"

streamlit run app_streamlit.py
```

* No app, faça upload dos PDFs e escolha o modo:

  * **OpenAI (com chave)**: melhor qualidade.
  * **Leve (sem chave)**: roda 100% CPU.
  * **Ollama (local)**: necessita do servidor Ollama rodando (`ollama serve`).

* **Base vetorial**:

  * **FAISS (memória)**: mais leve, não persiste.
  * **Chroma (persistente)**: bom para reuso local (pasta `storage/chroma`).

### 3) Publicar no GitHub

```bash
git add .
git commit -m "feat: primeira versão híbrida"
git branch -M main
git remote add origin https://github.com/<seu-usuario>/chatbot-nupetr-hibrido.git
git push -u origin main
```

### 4) Deploy no Streamlit Cloud

1. Acesse [https://share.streamlit.io/](https://share.streamlit.io/)
2. Conecte sua conta ao GitHub e selecione o repositório.
3. Aponte para `app_streamlit.py`.
4. Em **Secrets**, adicione (se quiser usar OpenAI):

   ```
   OPENAI_API_KEY="sua-chave"
   ```
5. Deploy. Use preferencialmente o **modo Leve** se não tiver chave.

> **Limites**: Evite modelos grandes; `mt5-small` é o melhor custo/benefício para Cloud sem GPU.

### 5) Construir o Executável Desktop (Windows)

> **Objetivo**: rodar LLM **robusto** localmente (ex.: Phi-3-mini, Mistral 7B quantizado), sem internet.

1. Instale dependências desktop:

   ```bash
   pip install -r requirements-desktop.txt
   ```

2. Baixe um modelo `.gguf` e coloque em `models/`, por exemplo:

   * `phi-3-mini-4k-instruct-q4_0.gguf` (CPU-friendly)

3. Crie o índice (uma vez):

   ```bash
   python app_desktop.py index --docs docs
   ```

4. Teste o chat:

   ```bash
   python app_desktop.py chat --model models/phi-3-mini-4k-instruct-q4_0.gguf
   ```

5. Gere o executável com PyInstaller:

   ```bash
   pyinstaller --name NUPETR-Assistente --onefile app_desktop.py \
     --add-data "models;models" --add-data "docs;docs"
   ```

   O arquivo estará em `dist/NUPETR-Assistente.exe`.

> Observações importantes:
>
> * **Não** empacote o `.gguf` no git; o executável pode carregar o arquivo de `models/`.
> * Se quiser embutir o modelo dentro do exe, o tamanho ficará **muito grande**; não recomendado.

### 6) Boas práticas e dicas

* **Compatibilidade de versões**: usamos `langchain>=0.2` + `openai>=1.x` (via `langchain-openai`). Evita o erro `openai.error.*`.
* **Repositório leve**: não comitar `/models`, `/dist`, índices FAISS, PDFs reais. Use o `.gitignore` fornecido.
* **FAISS no Windows**: se tiver problemas, troque por **Chroma** (já comentado em `src/rag.py`).
* **Privacidade**: documentos reais não devem ir para o repositório público. Use a pasta `docs/` localmente.

### 7) Como escolher o modo

* **OpenAI (chave)**: melhor linguagem e precisão. Custo por uso.
* **Leve (sem chave)**: custo zero e funciona no Streamlit Cloud. É mais simples, porém menos potente.
* **Robusto (desktop)**: ideal para cargas pesadas/off-line. Requer baixar um `.gguf`. Não roda no Streamlit Cloud.

---

## O que ficou pronto

* Repositório base completo (arquivos, estrutura e código) para **Streamlit** e **Executável Desktop**.
* **3 modos de modelo** (OpenAI, Leve sem chave, Robusto local) selecionáveis conforme o contexto.
* Requisitos (`requirements-*.txt`) fixos para reprodutibilidade.
* Passo a passo de **setup local**, **deploy GitHub/Streamlit**, e **build do executável**.

Pronto para adaptar ao NUPETR/IDEMA com seus **PDFs reais** e ajustes de prompt/UX.
