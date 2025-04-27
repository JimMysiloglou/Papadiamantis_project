import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from retriever import get_retrieved_documents



load_dotenv()

contextualize_instructions = "Χρησιμοποίησε τα παρακάτω αποσπάσματα κειμένου ως παράδειγμα:\n\n {context}."

prompt_with_context = ChatPromptTemplate.from_messages(
    [
        ("system", "Είσαι ένα συγγραφέας."),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("user", "{question}"),
        ("system", contextualize_instructions)
    ]
)

prompt_without_context = ChatPromptTemplate.from_messages([
    ("system", "Είσαι ένα συγγραφέας."),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("user", "{question}")
])

# Function to check if context exists
def context_exists(inputs: dict) -> bool:
    return bool(inputs.get("context"))

def initiate_memory(k=3):
    memory = ConversationBufferWindowMemory(k=k, return_messages=True)
    return memory


def generate_response(query, use_context, retrievers, source_type, memory, model, temperature):
    
    # Output parser
    output_parser = StrOutputParser()

    llm = ChatOpenAI(model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))

    # Choose prompt based on context presence
    dynamic_prompt = RunnableBranch(
    (context_exists, prompt_with_context),
    prompt_without_context
    )
    
    chain = (
    {
        "question": lambda x: x["question"],
        "context": RunnableLambda(get_retrieved_documents),
        "history": lambda x: memory.load_memory_variables({})["history"]
    }
    | dynamic_prompt
    | llm
    | output_parser
    )
    
    response = chain.invoke({'question': query, 'use_context': use_context, 'retrievers': retrievers, 'source_type':source_type})
    return response