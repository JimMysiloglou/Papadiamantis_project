import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import streamlit as st

# Best practice: store your credentials in environment variables
weaviate_url = st.secrets["WEAVIATE_URL"]
weaviate_api_key = st.secrets["WEAVIATE_API_KEY"]

@st.cache_resource
def connect_client(weaviate_url, weaviate_api_key):
    # Connect to Weaviate Cloud
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=st.secrets["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"]),
        skip_init_checks=True,
    )

@st.cache_resource
def create_embeddings(model='BAAI/bge-m3'):
    return HuggingFaceEmbeddings(model_name=model)

@st.cache_resource
def connect_client_cached():
    return connect_client(st.secrets["WEAVIATE_URL"], st.secrets["WEAVIATE_API_KEY"])

def create_vectorstore(client, embeddings, index_name):
    vectorstore = WeaviateVectorStore(client=client, embedding=embeddings,
                                  index_name=index_name, text_key='text')
    
    return vectorstore

def create_base_retriever(db, search_type='similarity', **search_kwargs):
    base_retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return base_retriever

def create_compression_retriever(base_retriever, compressor):
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever)
    return compression_retriever

def format_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def initiate_retrievers():
    weaviate_client = connect_client_cached()
    embeddings = create_embeddings()
    

    db = create_vectorstore(client=weaviate_client,
                            embeddings=embeddings,
                            index_name='PapadiamantisLangchain')
    print('Database created.')

    # Create four retrievers for each document type
    text_types = ["novels", "stories", "articles", "poems"]

    compressor = RankLLMRerank(top_n=3, model="gpt", gpt_model="gpt-4o-mini")

    compression_retrievers = {}
    for type in text_types:
        base_retriever = create_base_retriever(db, tenant=type, k=10)
        compression_retrievers[type] = create_compression_retriever(base_retriever, compressor)
        print(f'Compression retriever created for {type}')

    return compression_retrievers

@st.cache_resource
def get_retrievers():
    return initiate_retrievers()

def get_retrieved_documents(inputs):

    use_context = inputs.get("use_context")
    retrievers = inputs.get("retrievers", {})
    source_type = inputs.get("source_type", [])
    question = inputs.get("question")

    if not use_context or not source_type:
        return None
    
    if not question:
        raise ValueError("No question provided in inputs.")
    
    results = []
    for key in source_type:
        retriever = retrievers.get(key)
        if retriever is None:
            raise ValueError(f"Invalid source_type: {key}")
        docs = retriever.invoke(question)
        if docs:
            results.extend(docs)

    return format_context(results)