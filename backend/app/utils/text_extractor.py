import os
import PyPDF2
import docx

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from PDF, DOCX, or TXT files.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) == 0:
                     raise ValueError("PDF is empty")
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
                        
        elif ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        elif ext == '.txt':
            # Try utf-8, fall back to latin-1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    except Exception as e:
        # Re-raise the exception to be caught by the caller
        raise RuntimeError(f"Error parsing file: {str(e)}")

    if not text.strip():
        raise ValueError("File contains no extractable text.")
        
    return text.strip()
