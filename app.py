import streamlit as st
from retriever import initiate_retrievers
from query import initiate_memory, generate_response
import torch

torch.classes.__path__ = []

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

# Sidebar: Conversation History
st.sidebar.subheader("🕰️ Conversation History")

# Extract only user questions
user_questions = list(dict.fromkeys(
    m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"
))

selected_question = st.sidebar.selectbox(
    "Pick a previous question to re-ask:",
    options=user_questions,
    index=None,
    placeholder="Select a previous question..."
) if user_questions else None

# Sidebar:(outside the form!)
st.sidebar.divider()

# --- Sidebar Settings ---
with st.sidebar.form(key="settings_form"):
    st.sidebar.header("🔧 Settings")

    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
        ["gpt-4", "mistral-7b", "llama-3", "phi-2"],
        index=0
    )

    source_type = st.sidebar.radio("Select Source", ["all", "stories", "novels", "articles", "poems"])
    use_context = st.sidebar.checkbox("Use retrieved context", value=True)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_input = st.chat_input("Ask something to Papadiamantis ✍️")

user_input = None
if typed_input:
    user_input = typed_input
elif selected_question:
    user_input = selected_question

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
            model=llm_model,
            temperature=temperature
        )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)