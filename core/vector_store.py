from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def init_vector_store(embeddings: Optional[HuggingFaceEmbeddings], folder_path: str):
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=folder_path,
    )
    return vector_store
