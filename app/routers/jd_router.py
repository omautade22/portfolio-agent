from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

from app.services.extractor import extract_text
from app.services.vector_store_service import VectorStoreService

router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

API_KEY = os.getenv("CHAT_API_KEY")

UPLOAD_DIR = "./jd_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_keywords(text: str):
    # simple skill/keyword extraction
    important = []
    for word in text.split():
        if len(word) > 4:
            important.append(word.lower())
    return list(set(important))[:15]   # top keywords


@router.post("/match")
async def match_jd(x_api_key: str = Header(None)  ,file: UploadFile = File(...)):
    # Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    name = file.filename.lower()

    if not name.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF/DOCX supported")

    path = f"{UPLOAD_DIR}/{file.filename}"

    try:
        # save JD
        with open(path, "wb") as f:
            f.write(await file.read())

        # extract JD text
        jd_text = extract_text(path)

        # FAST SEARCH (no embedding JD)
        keywords = extract_keywords(jd_text)

        vector_store = VectorStoreService.get_instance()
        docs = vector_store.search(" ".join(keywords), k=8)

        profile = "\n".join([d.page_content for d in docs])

        prompt = f"""
You are a senior technical recruiter.

PROFILE:
{profile}

JOB DESCRIPTION:
{jd_text}

TASK:
1. Analyze skill match
2. Give percentage (0-100)
3. Classify:
   - Strong (80+)
   - Good (60-79)
   - Partial (40-59)
   - Weak (<40)

Format:

Match Percentage: <number>%
Verdict: <Strong/Good/Partial/Weak Match>

Why Om is a good fit:
- bullets

Skill gaps:
- bullets

End with:
"Om is always ready for challenges and would be excited to learn and upgrade the missing skills."
"""

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return {"analysis": res.choices[0].message.content}

    finally:
        if os.path.exists(path):
            os.remove(path)
