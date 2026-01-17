from pypdf import PdfReader
from docx import Document

def extract_text(file_path : str) -> str:
    if file_path.endswith(".pdf"):
        return extract_pdf(file_path)
    
    if file_path.endswith(".docx"):
        return extract_docx(file_path)
    
def extract_pdf(file_path : str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text(file_path)
        if page_text:
            text += page_text + "\n"

    return text

def extract_docx(file_path : str) -> str:
    doc = Document(file_path)
    text = ""

    for paragraph in doc.paragraphs():
        text += paragraph + "\n"

    return text