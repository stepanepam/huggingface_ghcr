import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from load import load_fairy_tale

from core.embedding import embeddings
from core.vector_store import init_vector_store
from settings.config import CHUNK_OVERLAP, CHUNK_SIZE, FAIRY_TALE_PATH, FOLDER_PATH


def fill_vector_store():
    text_fairy_tail = load_fairy_tale(FAIRY_TALE_PATH)

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = r_splitter.create_documents([text_fairy_tail])
    all_splits = r_splitter.split_documents(docs)

    vector_store = init_vector_store(embeddings, FOLDER_PATH)
    vector_store.add_documents(all_splits)
    print(vector_store._collection.count())


if __name__ == "__main__":
    fill_vector_store()
