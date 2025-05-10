from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Novel chapters - Use Recursive Splitter for Long Documents
novel_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", ".", "!", ";", "\n", "―"],
    keep_separator=False,
    length_function=num_tokens_from_string
)

# Short Stories - Use Recursive Splitter but with Smaller Chunks
short_story_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
    separators=["\n\n", ".", "!", ";", "\n", "―"],
    keep_separator=False,
    length_function=num_tokens_from_string
)

# Articles -Use Recursive Splitter but with Smaller Chunks
article_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=[".", "!", ";", "\n", "―"],
    keep_separator=False,
    length_function=num_tokens_from_string
)

# Poems - Use Recursive Splitter but with Smaller Chunks
hymn_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[".", "!", ";", "\n", "―"],
    keep_separator=False,
    length_function=num_tokens_from_string
)

def text_splitter(splitter, documents):
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            new_metadata = doc.metadata.copy()
            new_metadata['chunk'] = i
            split_docs.append(
                Document(
                    page_content=chunk,
                    metadata = new_metadata
                )
            )
    return split_docs
    

