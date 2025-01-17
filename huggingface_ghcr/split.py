from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Splitter:
    def __init__(self):
        self.text_splitter = self._create_text_splitter()

    def _create_text_splitter(self, chunk_size=200, chunk_overlap=30):
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return r_splitter

    @staticmethod
    def _load_fairy_tale(path_to_text: str = "Fairy Tales/Snow White.txt"):
        with open(path_to_text) as f:
            fairy_tale = f.read()
        return fairy_tale

    def _create_document(self):
        fairy_tale = self._load_fairy_tale()
        docs = self.text_splitter.create_documents([fairy_tale])
        return docs

    def split_page(self):
        docs = self._create_document()
        all_splits = self.text_splitter.split_documents(docs)
        return all_splits
