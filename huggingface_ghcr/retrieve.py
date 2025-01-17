from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_chroma import Chroma


def retriever(vectorstore: "Chroma"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever
