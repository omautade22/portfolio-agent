from fastapi import APIRouter, HTTPException, Header
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

from app.services.vector_store_service import VectorStoreService

router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
API_KEY = os.getenv("CHAT_API_KEY")


@router.post("/ask")
async def chat(
    query: str,
    x_api_key: str = Header(None)   # <-- THIS makes header visible in Swagger
):

    # Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    vector_store = VectorStoreService.get_instance()
    docs = vector_store.search(query, k=5)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You answer ONLY about Om Autade.
If unrelated say: I only answer about Om.

Context:
{context}

Question:
{query}

Answer:
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {"answer": res.choices[0].message.content}
