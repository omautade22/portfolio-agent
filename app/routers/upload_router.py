from fastapi import APIRouter, UploadFile, File, HTTPException, Header
import logging
import os

from app.services.extractor import extract_text
from app.services.chunking_service import ChunkingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Admin key
ADMIN_KEY = os.getenv("ADMIN_API_KEY")


def admin_guard(x_admin_key: str):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin access only")


@router.post("")
async def upload_profile(
    file: UploadFile = File(...),
    x_admin_key: str = Header(None)
):

    # Admin check
    admin_guard(x_admin_key)

    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF/DOCX allowed")

    path = f"{UPLOAD_DIR}/{file.filename}"

    try:
        # Save file
        with open(path, "wb") as f:
            f.write(await file.read())

        # Extract text
        text = extract_text(path)

        # Chunk
        chunker = ChunkingService.get_instance()
        chunks = chunker.chunk(text)

        # Store vectors
        vector_store = VectorStoreService.get_instance()
        vector_store.store(chunks)

        return {
            "message": "Profile ingested successfully",
            "chunks": len(chunks)
        }

    finally:
        # cleanup
        if os.path.exists(path):
            os.remove(path)
