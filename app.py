import streamlit as st
from retriever import initiate_retrievers
from query import initiate_memory, generate_response
import torch

torch.classes.__path__ = []

llm_display_names = {
    "ChatGPT 4.1": "gpt-4",
    "ChatGPT 4.1 nano": "gpt-4o-mini"
}

# --- Setup ---
st.set_page_config(page_title="Papadiamantis RAG Explorer", layout="wide")
st.title("📜 Papadiamantis RAG Explorer")

# --- Session State Initialization ---
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = initiate_retrievers()

if "memory" not in st.session_state:
    st.session_state["memory"] = initiate_memory()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset Conversation Button
if st.sidebar.button("🧹 New Conversation"):
    st.session_state.messages = []
    st.session_state.memory = initiate_memory()
    st.rerun()


if st.sidebar.button("Prepare download"):
    conversation = full_chat = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages)
    st.download_button(
        label="📥 Download Conversation",
        data=conversation,
        file_name="papadiamantis_chat.txt",
        on_click="ignore",
        type="primary"
    )

# --- Sidebar Settings ---
with st.sidebar.form(key="settings_form"):
    st.header("🔧 Settings")

    # Only set defaults if not already defined
    st.session_state.setdefault("llm_model", "ChatGPT 4.1 nano")
    st.session_state.setdefault("source_type", "all")
    st.session_state.setdefault("use_context", False)
    st.session_state.setdefault("temperature", 0.7)

    st.selectbox(
        "Select LLM Model",
        list(llm_display_names.keys()),
        key="llm_model"
    )

    st.radio(
        "Select Source",
        ["all", "stories", "novels", "articles", "poems"],
        key="source_type"
    )

    st.checkbox(
        "Use retrieved context",
        key="use_context"
    )

    st.slider(
        "Temperature", 0.0, 1.5, step=0.1, key="temperature"
    )

    st.form_submit_button("Apply Settings")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something to Papadiamantis ✍️")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = generate_response(
            query=user_input,
            use_context=st.session_state.use_context,
            retrievers=st.session_state["retrievers"],
            source_type=st.session_state.source_type,
            memory=st.session_state["memory"],
            model=llm_display_names[st.session_state.llm_model],
            temperature=st.session_state.temperature
        )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)