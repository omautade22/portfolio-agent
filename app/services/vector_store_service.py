from langchain_community.vectorstores import Chroma
import os
from app.services.embedding_service import HFEmbeddingService


class VectorStoreService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.persist_dir = "vector_db"
        os.makedirs(self.persist_dir, exist_ok=True)

        # IMPORTANT: pass OBJECT, NOT function
        self.embedder = HFEmbeddingService()

        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedder  # <-- THIS FIXES ERROR
        )

    def store(self, chunks):
        self.db.add_texts(chunks)
        self.db.persist()

    def search(self, query, k=5):
        return self.db.similarity_search(query, k=k)
