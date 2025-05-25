import streamlit as st
from retriever import get_retrievers
from query import initiate_memory, generate_response
import torch

torch.classes.__path__ = []

def prepare_download():
    lines = []
    counter = 1
    data = st.session_state.messages
    
    i = 0
    while i < len(data):
        if data[i]['role'] == 'user' and data[i+1]['role'] == 'assistant':
            user_text = data[i]['content'].strip()
            assistant_text = data[i+1]['content'].strip()
            settings = data[i+1]['settings']

            block = [f"{counter}. ________________________________________________"]
            block.append("🔧 Settings used:")
            block.append(f"    - Model: {settings['model']}")
            block.append(f"    - Temperature: {settings['temperature']}")
            block.append(f"    - Source Type: {settings['source_type']}")
            block.append(f"    - Context: {settings['context']}")
            if settings['system_prompt']:
                block.append(f"\nSYSTEM PROMPT:\n\"{settings['system_prompt']}\"")

            block.append(f"\nUSER:\n\"{user_text}\"")

            if settings['contextualization']:
                block.append(f"\nCONTEXTUALIZATION:\n\"{settings['contextualization']}\"")

            block.append(f"\nASSISTANT:\n\"{assistant_text}\"")
            block.append("______________________________________________________\n")

            lines.append("\n".join(block))
            counter += 1
            i += 2
        else:
            i+= 1

    return "\n\n".join(lines)

llm_display_names = {
    "ChatGPT 4.1": "gpt-4",
    "Gemini 2.0 flash": "gemini-2.0-flash",
    "Meltemi 7B": "https://ly8k72fbefhixvza.eu-west-1.aws.endpoints.huggingface.cloud",
    "Krikri 8B": "https://gxiojggyt022aqyt.eu-west-1.aws.endpoints.huggingface.cloud"
}

source_options = ["stories", "novels", "articles", "poems"]

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
    st.session_state.setdefault("llm_model", "ChatGPT 4.1")
    st.session_state.setdefault("use_context", False)
    st.session_state.setdefault("temperature", 0.5)
    st.session_state.setdefault("system_prompt", "")
    st.session_state.setdefault("contextualize_instructions", "")

    st.selectbox(
        "Select LLM Model",
        list(llm_display_names.keys()),
        key="llm_model"
    )

    selected_sources = st.pills(
        "Select Source Types (multiple allowed)",
        options=source_options,
        default=st.session_state.get("source_type", []),
        key="source_type",
        selection_mode="multi"
    )

    st.checkbox(
        "Use retrieved context",
        key="use_context"
    )

    st.slider(
        "Temperature", 0.0, 1.0, step=0.1, key="temperature"
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
    settings_log = {
    "model": st.session_state.llm_model,
    "temperature": st.session_state.temperature,
    "source_type": st.session_state.source_type,
    "context": bool(retrieved_context),
    "system_prompt": st.session_state.system_prompt.strip(),
    "contextualization": st.session_state.contextualize_instructions.strip()
    }
    
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