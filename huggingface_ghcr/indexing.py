import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


def indexing_page(splits: list[Document], folder_path: str = "./chroma_langchain_db"):
    embeddings = HuggingFaceEmbeddings(
        # api_key=inference_api_key,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    # embeddings = OpenAIEmbeddings()
    vector_store = init_vector_store(embeddings, folder_path)
    vector_store.add_documents(splits)
    print(vector_store._collection.count())
    return vector_store


def init_vector_store(embeddings: HuggingFaceEmbeddings, folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=folder_path,
    )
    return vector_store
