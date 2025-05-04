import streamlit as st
from retriever import get_retrievers
from query import initiate_memory, generate_response
import torch

torch.classes.__path__ = []

def prepare_download():
    lines = []
    for m in st.session_state.messages:
        role = m["role"].upper()
        text = m["content"]
        lines.append(f"{role}:\n{text}")
        if role == "ASSISTANT" and "settings" in m:
            lines.append(m["settings"])
    conversation = "\n\n".join(lines)
    return conversation

llm_display_names = {
    "ChatGPT 4.1": "gpt-4",
    "ChatGPT 4.1 nano": "gpt-4o-mini",
    "Gemini 2.0 flash": "gemini-2.0-flash",
    "Meltemi 7B": "https://ly8k72fbefhixvza.eu-west-1.aws.endpoints.huggingface.cloud",
    "Krikri 8B": "https://gxiojggyt022aqyt.eu-west-1.aws.endpoints.huggingface.cloud"
}

# --- Setup ---
st.set_page_config(page_title="Papadiamantis RAG Explorer", layout="wide")
st.title("📜 Papadiamantis RAG Explorer")

# --- Session State Initialization ---
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = get_retrievers()

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
    conversation = prepare_download()
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
    st.session_state.setdefault("system_prompt", "Είσαι ένα συγγραφέας.")
    st.session_state.setdefault("contextualize_instructions", "Χρησιμοποίησε τα παρακάτω αποσπάσματα κειμένου ως παράδειγμα:")

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

    system_prompt = st.text_area(
    "System Prompt (contextual role for the assistant)",
    key="system_prompt"
    )

    contextualize_instructions = st.text_area(
    "Contextualization Instructions (for RAG)",
    key="contextualize_instructions"
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
        response, retrieved_context = generate_response(
            query=user_input,
            use_context=st.session_state.use_context,
            retrievers=st.session_state["retrievers"],
            source_type=st.session_state.source_type,
            memory=st.session_state["memory"],
            model=llm_display_names[st.session_state.llm_model],
            temperature=st.session_state.temperature,
            system_prompt=st.session_state.system_prompt,
            contextualize_instructions=st.session_state.contextualize_instructions
        )
    
    # Build settings log
    settings_log = f"""
    🔧 Settings used:
    - Model: {st.session_state.llm_model}
    - Temperature: {st.session_state.temperature}
    - Source Type: {st.session_state.source_type}
    - Context: {"used" if retrieved_context else "not used"}
    - System Prompt: "{st.session_state.system_prompt}"
    - Contextualization: "{st.session_state.contextualize_instructions}"
    """.strip()

    # Store response and settings together
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "settings": settings_log
    })

    with st.chat_message("assistant"):
        st.markdown(response)

    if retrieved_context:
        with st.expander("📚 Retrieved Context Used"):
            st.markdown(f"```text\n{retrieved_context}\n```")
    st.caption("Context: " + ("used" if retrieved_context else "not used"))