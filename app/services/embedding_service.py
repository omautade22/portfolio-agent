from sentence_transformers import SentenceTransformer

class EmbeddingService:
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, texts : list[str]) -> list[list[float]]:
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()