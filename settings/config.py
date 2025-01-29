import os


CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
FOLDER_PATH = "./chroma_langchain_db"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FAIRY_TALE_PATH = "Fairy Tales/Snow White.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-l6-v2"
K_SEARCH_VECTOR_STORE = 6
