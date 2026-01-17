from fastapi import APIRouter, Request, HTTPException
from groq import Groq
import os
from dotenv import  load_dotenv
load_dotenv()

from app.services.vector_store_service import VectorStoreService

router = APIRouter(prefix="/chat")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

API_KEY = os.getenv("CHAT_API_KEY")


def validate(request: Request):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(403, "Unauthorized")


@router.post("/ask")
async def chat(request: Request, query: str):

    validate(request)

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
