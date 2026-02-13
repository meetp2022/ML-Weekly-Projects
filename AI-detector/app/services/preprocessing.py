"""Text preprocessing utilities."""
import re
from typing import List
import nltk

from app.core.logging import get_logger

logger = get_logger(__name__)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


def clean_text(text: str) -> str:
    """Clean and normalize text for scientific analysis."""
    # Standardize whitespace and remove control characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize quotes and unicode artifacts
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    
    return text.strip()


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    sentences = nltk.sent_tokenize(text)
    
    # Filter out very short sentences (likely noise)
    sentences = [s for s in sentences if len(s.split()) >= 3]
    
    return sentences


def preprocess_text(text: str) -> tuple[str, List[str]]:
    """Preprocess text for analysis."""
    cleaned = clean_text(text)
    sentences = tokenize_sentences(cleaned)
    return cleaned, sentences


def extract_stylometric_features(text: str, sentences: List[str]) -> Dict[str, float]:
    """Extract statistical markers of human vs AI writing style."""
    words = text.split()
    word_count = len(words)
    
    if word_count == 0 or not sentences:
        return {
            'avg_sentence_length': 0.0,
            'sentence_length_var': 0.0,
            'lexical_diversity': 0.0,
            'stopword_ratio': 0.0
        }

    # Sentence lengths
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lengths)
    sent_len_var = np.var(sent_lengths)
    
    # Lexical Diversity (Type-Token Ratio)
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    
    # Simple Stopword Ratio (common functional words)
    stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of'}
    stop_count = sum(1 for w in words if w.lower() in stopwords)
    stop_ratio = stop_count / word_count
    
    return {
        'avg_sentence_length': float(avg_sent_len),
        'sentence_length_var': float(sent_len_var),
        'lexical_diversity': float(ttr),
        'stopword_ratio': float(stop_ratio)
    }
