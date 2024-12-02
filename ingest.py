#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
import re
import json


from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from chromadb.config import Settings




# Load environment variables
persist_directory = 'chroma_kamy'
source_directory = 'source_data/individual_texts'
embeddings_model_name = 'all-MiniLM-L12-v2'
translate_bool = False
ner_bool = False
# text_split = os.environ.get('TEXT_SPLITTER')
chunk_size = 'full_document'
# chunk_overlap = int(os.environ.get('CHUNK_OVERLAP'))
cache_folder = '/local/work/baheryilmaz/.cache'

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def combine_and_save_translated_documents(documents: List[Document]):
    # Group documents by metadata
    grouped_documents = {}
    for doc in documents:
        metadata_tuple = tuple(sorted(doc.metadata.items()))  # Sort the items for consistent ordering
        if metadata_tuple not in grouped_documents:
            grouped_documents[metadata_tuple] = []
        grouped_documents[metadata_tuple].append(doc.page_content)

    # Create and save combined documents
    for metadata_tuple, contents in grouped_documents.items():
        combined_document = Document(
            page_content="\n".join(contents),
            metadata=dict(metadata_tuple)
        )

        # Generate a filename based on metadata or some unique identifier
        filename = f"{metadata_tuple}.txt"
        document_type = metadata_tuple[1][1].replace(' ', '_')
        document_id = metadata_tuple[0][1]
        document_date = metadata_tuple[2][1]
        filename = f"{document_type}_{document_id}_{document_date}.txt"

        # Save the combined document to a file
        with open('/local/work/kamarzideh/dataset/shipgpt/SHIP-GPT-Documents-English-debug/' +filename, "w", encoding="utf-8") as file:
            file.write(combined_document.page_content)


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    # if text_split == 'Sentence':
    #     text_splitter = CharacterTextSplitter(
    #         separator = ". ",
    #         chunk_size = chunk_size,
    #         chunk_overlap  = chunk_overlap,
    #         length_function = len,
    #     )
    #     #text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=['. ', '\n\n', '\n'])
    # else :
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    #texts = text_splitter.split_documents(documents)
    texts = documents # Directly use the documents
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    if translate_bool == 'True':
        texts = translate_docs(texts)
        combine_and_save_translated_documents(texts)
    if ner_bool == 'True':
        texts = create_ner(texts)
    for text in texts:
        #text.page_content = "passage: " + text.page_content
        text.page_content = re.sub('Seite \d+ von \d+', '', text.page_content)
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={"device": 0},
        cache_folder=cache_folder
    )

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        #texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        """
        document_names = [text.metadata['source'].replace('/home/kamarzideh/privateGPT/source_documents/', '').replace('.txt', '') for text in texts]
        regex_groups = [re.split("(\w+_)(\d+_)(\d{2}_\d{2}_\d{4})", name) for name in document_names]
        document_types = [regex[1] for regex in regex_groups]
        document_ids = [regex[2].replace('_', '') for regex in regex_groups]
        document_dates = [regex[3] for regex in regex_groups]
        metadata = {'document_type': document_types, 'document_id': document_ids, 'issued_date': document_dates}
        """
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
