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

# Sidebar:(outside the form!)
st.sidebar.divider()

# --- Sidebar Settings ---
with st.sidebar.form(key="settings_form"):
    st.sidebar.header("🔧 Settings")
    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
        list(llm_display_names.keys()),
        index=1
    )
    source_type = st.sidebar.radio("Select Source", ["all", "stories", "novels", "articles", "poems"])
    use_context = st.sidebar.checkbox("Use retrieved context", value=False)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

st.sidebar.divider()

if st.session_state.messages:
        full_chat = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages)
        st.download_button("📥 Download Conversation", full_chat, file_name="papadiamantis_chat.txt")

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
            use_context=use_context,
            retrievers=st.session_state["retrievers"],
            source_type=source_type,
            memory=st.session_state["memory"],
            model=llm_display_names[llm_model],
            temperature=temperature
        )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)