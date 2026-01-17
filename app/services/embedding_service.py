import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


class HFEmbeddingService:

    def __init__(self):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_API_KEY"),
        )

        # CONFIRMED WORKING MODEL
        self.model = "sentence-transformers/all-MiniLM-L6-v2"

    def embed_documents(self, texts: list[str]):
        embeddings = []

        for text in texts:
            vec = self.client.feature_extraction(
                text,
                model=self.model
            )
            embeddings.append(vec)

        return embeddings

    def embed_query(self, text: str):
        return self.client.feature_extraction(
            text,
            model=self.model
        )
