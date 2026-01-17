from fastapi import FastAPI, APIRouter
from app.routers.upload_router import router as upload_router
from app.routers.chat_router import router as chat_router
from app.routers.jd_router import router as jd_router
from dotenv import load_dotenv
load_dotenv() 

app = FastAPI(title="Portfolio agent")
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(jd_router, prefix="/jd", tags=["JD Match"])

@app.get("/")
def health():
    return {"status" : "running"}