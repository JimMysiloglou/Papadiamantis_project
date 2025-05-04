# 📜 Papadiamantis RAG Explorer

**Papadiamantis RAG Explorer** is a Streamlit app that allows users to chat with an AI inspired by Alexandros Papadiamantis' literary works. It uses Retrieval-Augmented Generation (RAG) and supports multiple LLMs, including OpenAI, Gemini, and Hugging Face endpoints. The context can be drawn from novels, stories, articles, and poems indexed in a Weaviate vector store.

## 🔍 Features

- 🧠 Retrieval-Augmented Generation (RAG) with compressed retrievers per document type.
- 📚 Query a specific source type or all together (novels, stories, articles, poems).
- 🤖 Chat with your preferred LLM: GPT-4, Gemini, Meltemi, Krikri.
- 🧩 Dynamic context injection.
- 🛠️ Customizable settings for temperature, prompt, and contextualization instructions.
- 💾 Option to download full conversation with settings used.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/papadiamantis-rag-explorer.git
cd papadiamantis-rag-explorer
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set up Secrets

Create a `.streamlit/secrets.toml` file and add your API keys:

```toml
WEAVIATE_URL = "https://your-weaviate-instance.weaviate.network"
WEAVIATE_API_KEY = "your_weaviate_api_key"

OPENAI_API_KEY = "your_openai_api_key"
GOOGLE_API_KEY = "your_google_api_key"
HF_TOKEN = "your_huggingface_token"
```

### 4. Run the App

```bash
streamlit run app.py
```

## 🧠 Architecture Overview

- `retriever.py`: Connects to Weaviate and builds contextual compression retrievers using RankLLM with `gpt-4o-mini`.
- `query.py`: Constructs dynamic prompts depending on whether context is available and queries the selected LLM.
- `app.py`: The main Streamlit app that manages UI, state, and logic.
- `requirements.txt`: Python dependencies for LangChain, Streamlit, LLM providers, and Weaviate.

## 📝 Citation Format

Context retrieved is presented with titles in Greek:
```text
Τίτλος: <Document Title>
<Document Content>
```

## 💡 Example Use Cases

- Compare different LLMs on how they mimic Papadiamantis' style.
- Study context relevance when using RAG with compressed vs raw document retrieval.
- Explore historical Greek literary texts interactively.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
