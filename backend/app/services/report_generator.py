from fpdf import FPDF
import os
from datetime import datetime
from app.models.schemas import DetectionResult

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Plagiarism Detection Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_report(result: DetectionResult, task_id: str) -> str:
    """
    Generates a PDF report for the detection result.
    Returns the path to the generated file.
    """
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Sanitize text function
    def clean_text(text):
        if not text: return ""
        # Replace common problematic characters
        replacements = {
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '-', '\u25e6': '-', '\u2022': '-'
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Force latin-1 compatibility
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Summary Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Report ID: {clean_text(task_id)}", 0, 1)
    pdf.cell(0, 10, f"Date: {clean_text(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Overall Plagiarism Score: {result.overall_similarity}%", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Verdict: {clean_text(result.verdict)}", 0, 1)
    pdf.multi_cell(0, 10, f"Explanation: {clean_text(result.explanation)}")
    pdf.ln(10)
    
    # Detailed Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Sentence Analysis:", 0, 1)
    pdf.set_font("Arial", size=11)
    
    for item in result.detailed_scores:
        pdf.set_text_color(0, 0, 0) # Reset to black
        similarity = item.similarity_score
        
        # Color code logic
        status_text = ""
        if similarity > 60:
            pdf.set_text_color(255, 0, 0) # Red
            status_text = "[HIGH]"
        elif similarity > 40:
            pdf.set_text_color(255, 165, 0) # Orange
            status_text = "[MED]"
        else:
             pdf.set_text_color(0, 128, 0) # Green
             status_text = "[LOW]"
             
        pdf.cell(0, 10, f"{status_text} Similarity: {similarity:.2f}%", 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, f"Sentence: {clean_text(item.sentence)}")
        if item.matched_source:
             pdf.set_font("Arial", 'I', 10)
             pdf.multi_cell(0, 8, f"Matches: {clean_text(item.matched_source)}")
             pdf.set_font("Arial", size=11)
        pdf.ln(5)

    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/report_{task_id}.pdf"
    pdf.output(report_path)
    
    return report_path
