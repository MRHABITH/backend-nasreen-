from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SentenceScore(BaseModel):
    sentence: str
    similarity_score: float
    matched_source: Optional[str] = None
    explanation: Optional[str] = None

class DetectionResult(BaseModel):
    overall_similarity: float
    detailed_scores: List[SentenceScore]
    verdict: str
    explanation: str
    metrics: Optional[Dict[str, Any]] = None

class TextCheckRequest(BaseModel):
    text: str
    sources: Optional[List[str]] = None

class RewriteRequest(BaseModel):
    text: str
    mode: str = "academic" # academic, humanize, fix
