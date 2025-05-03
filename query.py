from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from retriever import get_retrieved_documents
import streamlit as st

# Function to check if context exists
def context_exists(inputs: dict) -> bool:
    return bool(inputs.get("context"))

def initiate_memory(k=10):
    memory = ConversationBufferWindowMemory(k=k, return_messages=True)
    return memory

def generate_response(query, use_context, retrievers, source_type, memory, model, temperature, system_prompt, contextualize_instructions):

    prompt_with_context = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("user", "{question}"),
        ("system", contextualize_instructions + "\n\n{context}")
    ])

    prompt_without_context = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("user", "{question}")
    ])
    
    # Output parser
    output_parser = StrOutputParser()

    if model.startswith("gpt"):
        llm = ChatOpenAI(model=model,
                         temperature=temperature,
                         api_key=st.secrets["OPENAI_API_KEY"], 
                         max_tokens=None,
                         timeout=None,
                         max_retries=2)
    elif model.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(model=model,
                                     temperature=temperature,
                                     api_key=st.secrets["GOOGLE_API_KEY"],
                                     max_tokens=None,
                                     timeout=None,
                                     max_retries=2)
    else:
        endpoint = HuggingFaceEndpoint(
            endpoint_url=model,
            task="text-generation",
            max_tokens=None,
            temperature=temperature,
            huggingfacehub_api_token=st.secrets["HF_TOKEN"]
        )
        llm = ChatHuggingFace(llm=endpoint)

    # Select prompt based on whether context exists
    dynamic_prompt = RunnableBranch(
        (context_exists, prompt_with_context),
        prompt_without_context
    )

    # Fetch context if enabled
    context = ""
    if use_context:
        context = get_retrieved_documents({
            "question": query,
            "use_context": use_context,
            "retrievers": retrievers,
            "source_type": source_type
        })

    # Prepare input for the chain
    inputs = {
        "question": query,
        "context": context,
        "history": memory.load_memory_variables({})["history"]
    }

    # Compose the chain
    chain = dynamic_prompt | llm | output_parser

    # Invoke chain with precomputed inputs
    response = chain.invoke(inputs)

    return response, context