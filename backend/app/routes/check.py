from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional
import shutil
import uuid
import os
from app.services.nlp_engine import PlagiarismDetector
from app.models.schemas import DetectionResult, RewriteRequest
from app.store import results_store
from app.services.report_generator import generate_report

router = APIRouter()
detector = PlagiarismDetector()

@router.post("/check-text", response_model=dict)
def check_text(
    text: str = Form(...),
    sources: Optional[List[str]] = Form(None)
):
    """
    Check plagiarism for raw text against optional sources or internal dataset.
    """
    try:
        # Perform detection
        result = detector.detect(text, sources)
        
        # Generate Task ID
        task_id = str(uuid.uuid4())
        
        # Store result
        results_store[task_id] = result
        
        return {"task_id": task_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{task_id}", response_model=DetectionResult)
def get_results(task_id: str):
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return results_store[task_id]

@router.get("/download-report/{task_id}")
def download_report(task_id: str):
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = results_store[task_id]
    
    # Generate PDF
    # We use a background task or just generate on the fly if fast enough. 
    # PDF generation logic here is synchronous but fast enough for demo.
    
    try:
        report_path = generate_report(result, task_id)
        return FileResponse(report_path, media_type='application/pdf', filename=f"plagiarism_report_{task_id}.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.post("/upload-file")
def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, verify plagiarism, and return result.
    """
    # Generate unique filename to avoid collisions
    unique_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{unique_id}_{file.filename}"
    
    file_location = f"reports/{unique_filename}"
    os.makedirs("reports", exist_ok=True)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Extract text from file
    from app.utils.text_extractor import extract_text_from_file
    try:
        extracted_text = extract_text_from_file(file_location)
    except Exception as e:
        # Extract specific error message
        raise HTTPException(status_code=400, detail=str(e))
    
    if not extracted_text:
         raise HTTPException(status_code=400, detail="Could not extract text from file or file is empty.")

    # Run detection
    try:
        result = detector.detect(extracted_text)
        task_id = str(uuid.uuid4())
        results_store[task_id] = result
        
        return {"task_id": task_id, "result": result, "info": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rewrite", response_model=dict)
def rewrite_content(request: RewriteRequest):
    """
    Rewrite text using AI (Academic, Humanize, Fix).
    """
    rewritten_text = detector.rewrite_text(request.text, request.mode)
    return {"rewritten_text": rewritten_text}

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Rewrite / Plagiarism Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

@router.post("/generate-pdf")
def generate_pdf_endpoint(request: RewriteRequest):
    """
    Generates a PDF for the provided text.
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Handle unicode by replacing common non-latin chars or using a compatible font
    # For simplicity in this demo, we'll strip/replace incompatible chars or use latin-1
    text = request.text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 10, text)
    
    # Save to buffer or temp file
    temp_filename = f"reports/generated_{uuid.uuid4()}.pdf"
    os.makedirs("reports", exist_ok=True)
    pdf.output(temp_filename)
    
    return FileResponse(temp_filename, media_type='application/pdf', filename="ai_generated_document.pdf")
