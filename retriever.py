import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

load_dotenv()


# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

def connect_client(weaviate_url, weaviate_api_key):
    # Connect to Weaviate Cloud
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    print('Client status:', weaviate_client.is_ready())
    return weaviate_client

def create_embeddings(model='BAAI/bge-m3'):
    embeddings = HuggingFaceEmbeddings(model_name=model)
    print("Embedding function created.")
    return embeddings

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
    formated_documents = [f"Τίτλος: {doc.metadata['title']}\n{doc.page_content}" for doc in documents]
    return "\n\n".join(formated_documents)

def initiate_retrievers():
    weaviate_client = connect_client(weaviate_url, weaviate_api_key)
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
        base_retriever = create_base_retriever(db, search_type='mmr', tenant=type, k=10)
        compression_retrievers[type] = create_compression_retriever(base_retriever, compressor)
        print(f'Compression retriever created for {type}')

    return compression_retrievers

def get_retrieved_documents(inputs, retrievers, source_type):
    if not inputs.get("use_context", True):
        return None
    if source_type != "all":
        retriever = retrievers.get(source_type)
        docs = retriever.invoke(inputs["question"])
        return format_context(docs)
    else:
        results = []
        for _, retriever in retrievers.items():
            docs = retriever.invoke(inputs["question"])
            results.append(docs[0])
        return format_context(results)