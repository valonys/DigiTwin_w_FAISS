# 🚀 DigiTwin Streamlit App - ValLabs v3
import streamlit as st
import torch
import os
import time
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN")  # or hardcode "hf_xxxxxx"

# --- Streamlit page config ---
st.set_page_config(
    page_title="DigiTwin - ValLabs",
    page_icon="🚀",
    layout="centered"
)

# --- Display logo (placeholder) ---
st.image("https://github.com/valonys/DigiTwin/blob/fb79c7598b23f408f716222508788ea7cb45ab77/ValonyLabs_Logo.png", width=150)  # ⬅️ Replace with your actual logo path
st.title("🚀 DigiTwin - ValLabs 🚀")

# Avatars
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- Load model and tokenizer ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "amiguel/GM_Qwen1.8B_Finetune", 
        trust_remote_code=True, 
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "amiguel/GM_Qwen1.8B_Finetune",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    return model, tokenizer

model, tokenizer = load_model()

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DigiTwin intro system message ---
SYSTEM_PROMPT = (
    "You are DigiTwin, the digital twin of Ataliba, an inspection engineer with over 17 years of "
    "experience in mechanical integrity, reliability, piping, and asset management. "
    "Be precise, practical, and technical. Provide advice aligned with industry best practices."
)

# --- Build prompt using ChatML format ---
def build_prompt(messages):
    full_prompt = ""
    full_prompt += "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
    for msg in messages:
        role = msg["role"]
        if role == "user":
            full_prompt += "<|im_start|>user\n" + msg["content"] + "<|im_end|>\n"
        elif role == "assistant":
            full_prompt += "<|im_start|>assistant\n" + msg["content"] + "<|im_end|>\n"
    full_prompt += "<|im_start|>assistant\n"
    return full_prompt

# --- Generation function ---
def generate_response(prompt_text, model, tokenizer):
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "streamer": streamer
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer

# --- Display chat history ---
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask your inspection or reliability question..."):

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build prompt
    full_prompt = build_prompt(st.session_state.messages)

    if model and tokenizer:
        try:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                start_time = time.time()
                streamer = generate_response(full_prompt, model, tokenizer)

                response_container = st.empty()
                full_response = ""

                for chunk in streamer:
                    full_response += chunk
                    response_container.markdown(full_response + "▌", unsafe_allow_html=True)

                end_time = time.time()
                input_tokens = len(tokenizer(full_prompt)["input_ids"])
                output_tokens = len(tokenizer(full_response)["input_ids"])
                speed = output_tokens / (end_time - start_time)

                st.caption(
                    f"🔑 Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | "
                    f"🕒 Speed: {speed:.1f} tokens/sec"
                )

                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"⚡ Generation error: {str(e)}")
    else:
        st.error("🤖 Model not loaded!")
