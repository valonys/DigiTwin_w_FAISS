import streamlit as st
import os
import time
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from cerebras.cloud.sdk import Cerebras

# Load environment
load_dotenv()

# --- UI CONFIG & STYLE ---
st.set_page_config(page_title="DigiTwin RAG Forecast", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * {
        font-family: 'Tw Cen MT', sans-serif !important;
    }

    /* Sidebar arrow fix */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "▶";
        font-size: 1.3rem;
        margin-right: 0.4rem;
    }

    /* Top-right logo placement */
    .logo-container {
        position: fixed;
        top: 0.1rem;
        right: 0.4rem;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)

# Display logo (smaller, top-right)
st.markdown(
    """
    <div class="logo-container">
        <img src="https://github.com/valonys/DigiTwin/blob/29dd50da95bec35a5abdca4bdda1967f0e5efff6/ValonyLabs_Logo.png?raw=true" width="50">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 DigiTwin RAG Forecast App")

# --- AVATARS ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- SYSTEM PROMPTS ---
PROMPTS = {
    "Daily Report Summarization": """You are DigiTwin, an expert inspector and maintenance engineer...""",
    "5-Day Progress Report": """You are DigiTwin, an expert inspector with deep knowledge in KPIs, GM, CR...""",
    "Backlog Extraction": """You are DigiTwin, an expert inspector trained to extract and classify backlogs...""",
    "Inspector Expert": """You are DigiTwin, an expert inspector for advanced diagnosis and recommendation...""",
    "Complex Reasoning": """You are DigiTwin, trained to analyze multi-day reports using GS-OT-MIT-511 rules..."""
}

# --- STATE ---
for key in ["vectorstore", "chat_history", "model_intro_done", "current_model", "current_prompt"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None if key == "vectorstore" else False

# --- SIDEBAR ---
with st.sidebar:
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    uploaded_files = st.file_uploader("📄 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)
    prompt_type = st.selectbox("Choose Prompt Type", list(PROMPTS.keys()))

# --- PDF PARSING ---
def parse_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

@st.cache_resource
def build_faiss_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for i, doc in enumerate(_docs):
        for chunk in splitter.split_text(doc.page_content):
            chunks.append(LCDocument(page_content=chunk, metadata={"source": f"doc_{i}"}))
    return FAISS.from_documents(chunks, embeddings)

if uploaded_files:
    parsed_docs = [LCDocument(page_content=parse_pdf(f), metadata={"name": f.name}) for f in uploaded_files]
    st.session_state.vectorstore = build_faiss_vectorstore(parsed_docs)
    st.sidebar.success(f"{len(parsed_docs)} reports indexed.")

# --- RESPONSE LOGIC ---
def generate_response(prompt):
    messages = [{"role": "system", "content": PROMPTS[prompt_type]}]
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages.append({"role": "system", "content": f"Context from reports:\n{context}"})
    messages.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        if model_alias == "EE Smartest Agent":
            client = openai.OpenAI(api_key=os.getenv("API_KEY"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(model="grok-3", messages=messages, stream=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"

        elif model_alias == "JI Divine Agent":
            client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.sambanova.ai/v1")
            response = client.chat.completions.create(model="DeepSeek-R1-Distill-Llama-70B", messages=messages, stream=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"

        elif model_alias == "EdJa-Valonys":
            client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
            response = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=messages)
            content = response.choices[0].message.content if hasattr(response.choices[0], "message") else str(response.choices[0])
            for word in content.split():
                full_response += word + " "
                yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                time.sleep(0.01)

        elif model_alias == "XAI Inspector":
            model_id = "amiguel/GM_Qwen1.8B_Finetune"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", token=os.getenv("HF_TOKEN"))
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
            output = model.generate(input_ids, max_new_tokens=512, do_sample=True, top_p=0.9)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

        elif model_alias == "Valonys Llama":
            model_id = "amiguel/Llama3_8B_Instruct_FP16"
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=os.getenv("HF_TOKEN"))
            input_ids = tokenizer(PROMPTS[prompt_type] + "\n\n" + prompt, return_tensors="pt").to(model.device)
            output = model.generate(**input_ids, max_new_tokens=512)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

    except Exception as e:
        yield f"<span style='color:red'>⚠️ Error: {str(e)}</span>"

# --- AGENT INTRO ---
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias or st.session_state.current_prompt != prompt_type:
    agent_intros = {
        "EE Smartest Agent": "💡 EE Agent Activated — Pragmatic & Smart",
        "JI Divine Agent": "✨ JI Agent Activated — DeepSeek Reasoning",
        "EdJa-Valonys": "⚡ EdJa Agent Activated — Cerebras Speed",
        "XAI Inspector": "🔍 XAI Inspector — Qwen Custom Fine-tune",
        "Valonys Llama": "🦙 Valonys Llama — LLaMA3-Based Reasoning"
    }
    st.session_state.chat_history.append({"role": "assistant", "content": agent_intros.get(model_alias)})
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias
    st.session_state.current_prompt = prompt_type

# --- RENDER HISTORY ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask a summary or forecast about the reports..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        response_placeholder.markdown(full_response, unsafe_allow_html=True)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
