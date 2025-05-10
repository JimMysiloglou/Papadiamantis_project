import os
import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from collections import defaultdict
from tqdm import tqdm

def num_tokens_from_string(string: str, encoding_name: str='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def extract_text(file_path):
    """Extracts text from a file, removing duplicate titles, trimming whitespace,
       removing footnotes that start with a number, and reducing excessive whitespace.
       Also extracts the publication year if present in the title and removes brackets from the title."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]

        if not lines:
            raise ValueError(f"File {file_path} is empty.")

        title = lines.pop(0)

        # Extract publication year if present in the title (including cases where it is in parentheses)
        year_match = re.search(r'\(?(\d{4}\b)\)?', title)
        publication_year = int(year_match.group(1)) if year_match else None
        title = re.sub(r'\s*\(?(\d{4}\b)\)?\s*', '', title).strip()  # Remove year and parentheses from title

        # Remove square brackets from the title
        title = re.sub(r'\[|\]', '', title)

        # Remove second occurrence of title if it appears after 2 or 3 newlines
        while lines and lines[0] == "":
            lines.pop(0)  # Remove empty lines

        if lines and lines[0] == title:
            lines.pop(0)  # Remove duplicate title

        # Remove footnotes (lines starting with a number)
        main_text_lines = []
        for line in lines:
            if re.match(r'^\d+\.', line):
                break  # Stop at the first footnote
            main_text_lines.append(line)

        main_text = "\n".join(main_text_lines)

        # Normalize multiple blank lines to exactly one
        main_text = re.sub(r'\n\s*\n+', '\n\n', main_text)

        # Strip any leading/trailing whitespace from the whole text
        main_text = main_text.strip()

        # Remove '[', ']', and '*' from the text
        main_text = re.sub(r'[\[\]\*]', '', main_text)

        # Remove occurrences of the pattern ". ."
        main_text = re.sub(r'\. \.', '', main_text)

    return title, main_text, publication_year
    
def split_by_chapters(text):
    # Adjust pattern based on the novel’s formatting
    chapter_pattern = r"(ΚΕΦΑΛΑΙΟΝ\s+[Α-Ω]{1,2}´\s*-\s*.+|ΠΡΟΛΟΓΟΣ|ΕΠΙΛΟΓΟΣ)"  # Detects "ΚΕΦΑΛΑΙΟΝ Α’" and "ΠΡΟΛΟΓΟΣ"
    chapters = re.split(chapter_pattern, text)

    # Merge chapter titles with corresponding text
    structured_chapters = []
    if chapters[0]: # Checking for text before prologue
        title = 'ΕΙΣΑΓΩΓΗ'
        content = chapters[0]
        structured_chapters.append((title, content))
    for i in range(1, len(chapters), 2):  # Skip first empty split
        title = chapters[i].strip()  # Chapter title
        content = chapters[i + 1].strip() if (i + 1) < len(chapters) else ""
        if content:
            structured_chapters.append((title, content))

    return structured_chapters
    
def directory_to_documents(directory, split_chapters=True):
    """Creates a list of Document objects, each containing the text (title + main_text)
       and using the folder paths as metadata."""
    documents = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        folder_key = relative_path if relative_path != "." else os.path.basename(directory)
        folder_parts = folder_key.split(os.sep)
        metadata = {"type": folder_parts[0]} if folder_parts else {}
        if len(folder_parts) > 1:
            metadata["theme"] = folder_parts[1]
        else:
            metadata["theme"] = "Άγνωστη"


        for file in tqdm(files):
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                title, main_text, publication_year = extract_text(file_path)
                file_metadata = metadata.copy()
                file_metadata["title"] = title
                file_metadata["year"] = publication_year if publication_year else 0
                if split_chapters:
                    if file_metadata['type'].strip().lower() == 'μυθιστορήματα' and main_text.strip():
                        chapters = split_by_chapters(main_text)
                        for chapter, text in chapters:
                            file_metadata["chapter"] = chapter
                            doc = Document(page_content=text, metadata=file_metadata)
                            documents[file_metadata['type']].append(doc)
                    else:
                        if main_text.strip():
                            file_metadata['chapter'] = 'Not applied'
                            doc = Document(page_content=main_text, metadata=file_metadata)
                            documents[file_metadata['type']].append(doc)
                else:
                    if main_text.strip():
                        doc = Document(page_content=main_text, metadata=file_metadata)
                        documents[file_metadata['type']].append(doc)


    return documents
 
