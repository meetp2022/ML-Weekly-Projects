import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

def sentence_splitter(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def create_sentence_chunks(documents, sentences_per_chunk=3):
    chunks = []
    for doc in documents:
        sentences = sentence_splitter(doc.page_content)
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_text = '. '.join(sentences[i:i+sentences_per_chunk]) + '.'
            if len(chunk_text) > 20:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=doc.metadata.copy()
                    )
                )
    return chunks

def get_chunkers(chunk_size=300, overlap=50):
    fixed = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    recursive = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    return fixed, recursive
