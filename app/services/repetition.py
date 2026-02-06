"""Repetition detection metrics."""
from typing import List
from collections import Counter
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def calculate_ngram_repetition(text: str, n: int = 3) -> float:
    """Calculate n-gram repetition ratio.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Repetition ratio (0-1, higher = more repetitive)
    """
    words = text.lower().split()
    
    if len(words) < n:
        return 0.0
    
    # Generate n-grams
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    # Count occurrences
    ngram_counts = Counter(ngrams)
    
    # Calculate repetition: (total - unique) / total
    total_ngrams = len(ngrams)
    unique_ngrams = len(ngram_counts)
    
    repetition = (total_ngrams - unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0
    
    return float(repetition)


def calculate_token_diversity(text: str) -> float:
    """Calculate token diversity (unique tokens / total tokens).
    
    Args:
        text: Input text
        
    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    words = text.lower().split()
    
    if not words:
        return 1.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    diversity = unique_words / total_words
    
    return float(diversity)


def calculate_repetition_score(text: str) -> float:
    """Calculate overall repetition score.
    
    Combines n-gram repetition and token diversity.
    
    Args:
        text: Input text
        
    Returns:
        Repetition score (0-1, higher = more repetitive/AI-like)
    """
    # Calculate metrics
    bigram_rep = calculate_ngram_repetition(text, n=2)
    trigram_rep = calculate_ngram_repetition(text, n=3)
    diversity = calculate_token_diversity(text)
    
    # Combine metrics
    # Higher repetition and lower diversity = more AI-like
    repetition_score = (bigram_rep * 0.3 + trigram_rep * 0.3 + (1 - diversity) * 0.4)
    
    logger.debug(
        f"Repetition: {repetition_score:.3f} "
        f"(bigram={bigram_rep:.3f}, trigram={trigram_rep:.3f}, diversity={diversity:.3f})"
    )
    
    return float(repetition_score)


def normalize_repetition(repetition: float) -> float:
    """Normalize repetition to AI score (0-100).
    
    Args:
        repetition: Raw repetition score (0-1)
        
    Returns:
        Normalized score (0-100, higher = more AI-like)
    """
    normalized = 100 * repetition
    
    return float(normalized)
