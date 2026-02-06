"""Burstiness metrics for sentence length variation."""
from typing import List
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def calculate_burstiness(sentences: List[str]) -> float:
    """Calculate burstiness score based on sentence length variation.
    
    Human writing tends to have more varied sentence lengths (higher burstiness).
    AI writing tends to have more uniform sentence lengths (lower burstiness).
    
    Args:
        sentences: List of sentences
        
    Returns:
        Burstiness score (0-1, higher = more human-like)
    """
    if len(sentences) < 2:
        return 0.5  # Neutral score for very short texts
    
    # Calculate sentence lengths (in words)
    lengths = [len(sentence.split()) for sentence in sentences]
    
    # Calculate standard deviation
    std_dev = np.std(lengths)
    mean_length = np.mean(lengths)
    
    # Coefficient of variation (CV)
    if mean_length > 0:
        cv = std_dev / mean_length
    else:
        cv = 0.0
    
    # Normalize to 0-1 range (typical CV range is 0-1 for text)
    burstiness = np.clip(cv, 0.0, 1.0)
    
    logger.debug(f"Burstiness: {burstiness:.3f} (std={std_dev:.2f}, mean={mean_length:.2f})")
    
    return float(burstiness)


def normalize_burstiness(burstiness: float) -> float:
    """Normalize burstiness to AI score (0-100).
    
    Args:
        burstiness: Raw burstiness score (0-1)
        
    Returns:
        Normalized score (0-100, higher = more AI-like)
    """
    # Invert: lower burstiness = higher AI score
    normalized = 100 * (1 - burstiness)
    
    return float(normalized)
