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
    """Clean and normalize text.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


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
    """Preprocess text for analysis.
    
    Args:
        text: Raw input text
        
    Returns:
        Tuple of (cleaned_text, sentences)
    """
    cleaned = clean_text(text)
    sentences = tokenize_sentences(cleaned)
    
    logger.debug(f"Preprocessed text: {len(sentences)} sentences")
    
    return cleaned, sentences
