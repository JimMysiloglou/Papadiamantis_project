import streamlit as st
from retriever import get_retrieved_documents
from query import generate_response

# Placeholder imports — make sure your functions use these selections
# from your_project import generate_response, get_retrieved_documents

# Set up Streamlit page
st.set_page_config(page_title="Papadiamantis RAG", layout="wide")
st.title("📜 Papadiamantis RAG Explorer")

# --- Sidebar Settings ---
st.sidebar.header("🔧 Settings")

# 1. LLM selection
llm_model = st.sidebar.selectbox(
    "Select LLM Model",
    ["GPT-4", "Gemini", "Meltemi", "Krikri"],
    index=0
)

# 2. Source type selection
source_type = st.sidebar.radio("Choose content type", ["All", "Short Stories", "Novels", "Articles", "Poems", "None"])

# 3. Temperature slider
temperature = st.sidebar.slider("Generation temperature", 0.0, 1.5, 0.7, 0.1)

# --- Main Area ---
query = st.text_input("Ask a question or request a story:")

if query:
    with st.spinner(f"Using {llm_model} to retrieve and generate..."):
        result = generate_response(
            query=query,
            source_type=source_type,
            temperature=temperature,
            llm=llm_model  # Make sure your backend handles this
        )
        retrieved = get_retrieved_documents(query, source_type)

    st.subheader("📝 Generated Output")
    st.write(result)

    with st.expander("📚 Retrieved Contexts"):
        for doc in retrieved:
            st.markdown(f"**{doc['source']}**: {doc['content']}")

    st.download_button("📥 Download result", result, file_name="papadiamantis_output.txt")
