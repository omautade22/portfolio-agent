from fastapi import APIRouter, UploadFile, File, HTTPException
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
from app.services.extractor import extract_text
from app.services.vector_store_service import VectorStoreService

router = APIRouter(prefix="/jd")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

UPLOAD_DIR = "./jd_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/match")
async def match_jd(file: UploadFile = File(...)):

    name = file.filename.lower()

    if not name.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF/DOCX")

    path = f"{UPLOAD_DIR}/{file.filename}"

    try:
        with open(path, "wb") as f:
            f.write(await file.read())

        jd_text = extract_text(path)

        vector_store = VectorStoreService.get_instance()
        docs = vector_store.search(jd_text, k=8)

        profile = "\n".join([d.page_content for d in docs])

        prompt = f"""
You are a senior recruiter.

PROFILE:
{profile}

JD:
{jd_text}

Give:

Match Percentage: %
Verdict:
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


if __name__ == "__main__":
    key = os.getenv("GROQ_API_KEY")
    print(key)