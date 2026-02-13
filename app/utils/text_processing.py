import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercasing, removing extra spaces.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text: str) -> str:
    """
    Advanced preprocessing: removing punctuation, stopwords.
    Returns a cleaned string ready for TF-IDF or embedding.
    """
    text = clean_text(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(filtered_tokens)

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using NLTK.
    """
    return sent_tokenize(text)
