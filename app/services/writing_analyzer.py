import textstat
from spellchecker import SpellChecker
from typing import Dict, List, Any
import re

class WritingAnalyzer:
    def __init__(self):
        self.spell = SpellChecker()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for readability, spelling, and basic statistics.
        """
        if not text.strip():
            return {
                "readability_score": 0,
                "readability_label": "N/A",
                "spelling_errors": 0,
                "grammar_issues": 0, # Placeholder for more complex grammar
                "word_count": 0,
                "sentence_count": 0,
                "suggestion": "No text provided."
            }

        # 1. Readability (Flesch Reading Ease)
        # 90-100: Very Easy, 0-30: Very Confusing
        score = textstat.flesch_reading_ease(text)
        label = self._get_readability_label(score)

        # 2. Spelling
        # Find words
        words = re.findall(r'\b\w+\b', text.lower())
        # Spell check (this can be slow for very large texts, consider optimizing if needed)
        misspelled = self.spell.unknown(words)
        spelling_count = len(misspelled)

        # 3. Basic Stats
        sentence_count = textstat.sentence_count(text)
        word_count = len(words)

        # 4. "Grammar" / Style Heuristics (Simple proxies)
        # e.g., sentences that are too long (> 30 words)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        long_sentences = sum(1 for s in sentences if len(s.split()) > 30)
        
        # Heuristic score for grammar/style based on long sentences + spelling density
        grammar_issues = long_sentences + (spelling_count // 5) 

        return {
            "readability_score": score,
            "readability_label": label,
            "spelling_errors": spelling_count,
            "grammar_issues": grammar_issues,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "additional_issues": long_sentences # e.g. "Conciseness"
        }

    def _get_readability_label(self, score):
        if score > 90: return "Very Easy"
        if score > 80: return "Easy"
        if score > 70: return "Fairly Easy"
        if score > 60: return "Standard"
        if score > 50: return "Fairly Difficult"
        if score > 30: return "Difficult"
        return "Very Confusing"
