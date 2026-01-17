from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


class VectorStoreService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.persist_dir = "vector_db"
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def store(self, chunks):
        self.db.add_texts(chunks)
        self.db.persist()

    def search(self, query, k=5):
        return self.db.similarity_search(query, k=k)