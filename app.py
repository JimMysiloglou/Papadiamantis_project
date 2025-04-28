import streamlit as st
from retriever import initiate_retrievers
from query import initiate_memory, generate_response
import torch

torch.classes.__path__ = []

# --- Setup ---
st.set_page_config(page_title="Papadiamantis RAG Explorer", layout="wide")
st.title("📜 Papadiamantis RAG Explorer")

# --- Sidebar Settings ---
st.sidebar.header("🔧 Settings")

llm_model = st.sidebar.selectbox(
    "Select LLM Model",
    ["gpt-4", "mistral-7b", "llama-3", "phi-2"],
    index=0
)

source_type = st.sidebar.radio("Select Source", ["all", "stories", "novels", "articles", "poems"])
use_context = st.sidebar.checkbox("Use retrieved context", value=True)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

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

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask something the model ✍️")

if user_input:

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant reply
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

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
        
    with st.chat_message("assistant"):
        st.markdown(response)
