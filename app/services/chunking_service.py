from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

    def chunk(self, text: str):
        docs = self.splitter.create_documents([text])
        return [d.page_content for d in docs]
