import os
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()

# Suppress TensorFlow and HF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import numpy as np
from app.models.schemas import DetectionResult, SentenceScore
from app.utils.text_processing import split_into_sentences
from app.services.writing_analyzer import WritingAnalyzer
from groq import Groq
import json

import requests
from bs4 import BeautifulSoup
from googlesearch import search

class PlagiarismDetector:
    def __init__(self):
        print("Loading Hybrid NLP Pipeline...")
        if not os.getenv("HF_TOKEN"):
            print("Warning: HF_TOKEN not found. Set it in backend/.env to enable authenticated requests.")
        
        # --- Component 1: Embedding Model ---
        # Switch to 'all-MiniLM-L6-v2' to fix Memory Error and improve speed
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        # Load Writing Analyzer
        self.writing_analyzer = WritingAnalyzer()

        # Initialize Groq Client (LLM Component)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = None
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("Groq AI Client Initialized (Llama-3.3-70b) for Deep Analysis")
            except Exception as e:
                print(f"Failed to init Groq: {e}")
        else:
            print("Warning: GROQ_API_KEY not found. Advanced AI verification disabled.")

        # Load internal dataset (dummy for now)
        self.internal_corpus = [
            "Plagiarism is the representation of another author's language, thoughts, ideas, or expressions as one's own original work.",
            "The rapid growth of the Internet has made it easier to plagiarize content from various sources.",
            "Natural Language Processing enables computers to understand specific aspects of human language."
        ]
        self.internal_embeddings = self.model.encode(self.internal_corpus, convert_to_tensor=True)
        print("Hybrid NLP Engine Ready.")

    def _web_search(self, text: str) -> List[str]:
        """
        Searches the web for sentences from the text to find potential sources.
        Returns a list of scraped text content from top URLs.
        """
        print("Performing Web Search...")
        found_texts = []
        sentences = split_into_sentences(text)
        
        # Select up to 2 representative sentences to search (beginning and middle)
        search_queries = []
        if len(sentences) > 0: search_queries.append(sentences[0])
        if len(sentences) > 5: search_queries.append(sentences[len(sentences)//2])
        
        urls = set()
        for query in search_queries:
            try:
                # Search Google for the exact phrase
                for url in search(query, num_results=3, lang="en"):
                     urls.add(url)
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")

        print(f"Found {len(urls)} potential sources: {list(urls)}")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract paragraphs
                    paragraphs = [p.get_text() for p in soup.find_all('p')]
                    page_text = " ".join(paragraphs)
                    if len(page_text) > 100:
                        found_texts.append(page_text[:5000]) # Limit context per page
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
        
        return found_texts

    def detect(self, text: str, sources: Optional[List[str]] = None) -> DetectionResult:
        """
        Executes the Hybrid Pipeline:
        1. Embedding Retrieval
        2. Similarity Scoring (Hybrid: Vector + Lexical)
        3. Threshold Logic
        4. LLM Analysis
        """
        sentences = split_into_sentences(text)
        detailed_scores = []
        
        # --- Step 0: Pre-Analysis ---
        writing_stats = self.writing_analyzer.analyze(text)

        # Prepare Corpus
        comparison_sentences = []
        if sources:
            for src in sources:
                comparison_sentences.extend(split_into_sentences(src))
        else:
            # Inject Web Search Results if no sources provided
            print("No sources provided. Initiating Web Search...")
            web_sources = self._web_search(text)
            if web_sources:
                 for src in web_sources:
                     comparison_sentences.extend(split_into_sentences(src))
            
            # Reset to internal if still empty (fallback)
            if not comparison_sentences:
                comparison_sentences = self.internal_corpus
        
        if not comparison_sentences:
           comparison_sentences = [""]

        # --- Step 1: Embedding Retrieval ---
        corpus_embeddings = self.model.encode(comparison_sentences, convert_to_tensor=True)
        input_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Calculate Vector Similarity (Cosine)
        cosine_scores = util.cos_sim(input_embeddings, corpus_embeddings)

        # --- Step 2 & 3: Similarity Scoring & Threshold Logic ---
        results = []
        for idx, sent in enumerate(sentences):
            if len(sent.strip()) < 10:
                results.append(self._create_score(sent, 0.0, None, "Too short."))
                continue

            # Best Semantic Match
            best_idx = np.argmax(cosine_scores[idx].cpu().numpy())
            semantic_score = cosine_scores[idx][best_idx].item() * 100
            matched_sent = comparison_sentences[best_idx]
            
            # Exact Match (Lexical validation)
            exact_score = self._get_exact_match_score(sent, matched_sent)
            
            # Hybrid Score Strategy
            # Boost score if web search found something highly relevant
            final_score = max(semantic_score, exact_score)
            
            # Threshold Logic
            explanation = self._determine_verdict(final_score, semantic_score, exact_score)
            
            results.append({
                "sentence": sent,
                "score": final_score,
                "match": matched_sent if final_score > 45 else None, # Higher threshold for showing match
                "explanation": explanation
            })

        # --- Step 4: LLM Analysis (Deep Verification) ---
        # Select sentences that are "ambiguous" (e.g. 50-90% similarity) for AI review
        detailed_scores = []
        for res in results:
            final_score = res["score"]
            explanation = res["explanation"]
            
            if self.groq_client and 50 < final_score < 95:
                # LLM Verification for ambiguous cases
                ai_result = self._verify_with_groq(res["sentence"], res["match"])
                if ai_result:
                    # Update with AI insights
                    final_score = ai_result.get("score", final_score)
                    explanation = f"[AI Verified] {ai_result.get('explanation', explanation)}"
            
            detailed_scores.append(SentenceScore(
                sentence=res["sentence"],
                similarity_score=final_score,
                matched_source=res["match"],
                explanation=explanation
            ))

        # Final Aggregation
        overall_similarity = self._calculate_overall(detailed_scores)
        
        return DetectionResult(
            overall_similarity=round(overall_similarity, 2),
            detailed_scores=detailed_scores,
            verdict="Plagiarism Detected" if overall_similarity > 15 else "Clean",
            explanation=f"Hybrid Analysis Complete (Web Search Active). Readability: {writing_stats['readability_label']}.",
            metrics=writing_stats 
        )

    def _get_exact_match_score(self, sent1, sent2):
        """Calculates Jaccard/N-Gram overlap for lexical similarity."""
        set1 = set(sent1.lower().split())
        set2 = set(sent2.lower().split())
        if not set1 or not set2: return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return (intersection / union) * 100

    def _determine_verdict(self, final_score, semantic_score, exact_score):
        """Applies Threshold Logic to determine the initial explanation."""
        if final_score > 90: return "Identical match (High Confidence)."
        if final_score > 75: return "Highly similar content."
        if final_score > 60: return "Likely paraphrased."
        if final_score > 40: return "Possible similarity."
        return "Original content."

    def _calculate_overall(self, scores: List[SentenceScore]) -> float:
        """Calculates the aggregate plagiarism score."""
        if not scores: return 0.0
        flagged = [s for s in scores if s.similarity_score > 60]
        return (len(flagged) / len(scores)) * 100

    def _verify_with_groq(self, suspect_text: str, source_text: str) -> dict:
        """
        Uses Llama-3 via Groq to verify plagiarism and generate an explanation.
        """
        if not self.groq_client: return None

        prompt = f"""
        Analyze these two text segments for plagiarism.
        
        Text A (Suspect): "{suspect_text}"
        Text B (Source): "{source_text}"

        Task:
        1. Compare meaning, structure, and vocabulary.
        2. Determine if Text A is derived from Text B.
        
        Return ONLY a JSON object with:
        - "is_plagiarism": boolean
        - "score": number (0-100) representing similarity
        - "explanation": short, one-sentence explanation of *why*.
        """

        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception:
            return None

    def _create_score(self, sent, score, match, explanation):
        return {
            "sentence": sent,
            "score": score,
            "match": match,
            "explanation": explanation
        }

    def rewrite_text(self, text: str, mode: str = "academic") -> str:
        """
        Rewrites text using Llama-3 based on the selected mode.
        """
        if not self.groq_client:
            return "Error: Groq AI is not enabled. Please check your API Key."

        prompts = {
            "academic": """
            Rewrite the following text to reduce plagiarism score to near zero.
            - Change structure and vocabulary.
            - Maintain facts.
            - Use academic tone.
            Return ONLY the rewritten text.
            """,
            "humanize": """
            Refine the text to sound natural and human-like.
            - Vary sentence length.
            - Remove robotic flow.
            Return ONLY the rewritten text.
            """,
            "fix": """
            Fix grammar and flow.
            Return ONLY the corrected text.
            """,
            "comprehensive": """
            Act as a professional editor. 
            1. Fix ALL grammar, spelling, and punctuation errors.
            2. Improve sentence flow and clarity.
            3. Maintain the original meaning but make it sound professional and polished.
            Return ONLY the rewritten text.
            """
        }

        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": prompts.get(mode, prompts["academic"])},
                    {"role": "user", "content": f"Text:\n{text}"}
                ],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error during rewriting: {str(e)}"

