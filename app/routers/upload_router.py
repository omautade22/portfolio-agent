from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
import os

from app.services.extractor import extract_text
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/profile")

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize services ONCE
chunker = ChunkingService()
embedder = EmbeddingService()
vector_store = VectorStoreService()


@router.post("/upload")
async def upload_profile(file: UploadFile = File(...)):

    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF/DOCX allowed")

    path = f"{UPLOAD_DIR}/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    text = extract_text(path)

    chunker = ChunkingService.get_instance()
    chunks = chunker.chunk(text)

    vector_store = VectorStoreService.get_instance()
    vector_store.store(chunks)

    return {
        "message": "Profile ingested successfully",
        "chunks": len(chunks)
    }