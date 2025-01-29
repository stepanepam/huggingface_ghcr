from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_chroma import Chroma

from settings.config import K_SEARCH_VECTOR_STORE


def get_retriever(vectorstore: "Chroma"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_SEARCH_VECTOR_STORE})
    return retriever
