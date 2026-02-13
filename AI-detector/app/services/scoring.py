"""Scoring and aggregation logic."""
from typing import List, Dict
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.services.perplexity import (
    calculate_perplexity,
    calculate_sentence_perplexities,
    normalize_perplexity,
    calculate_perplexity_variance,
    normalize_variance
)
from app.services.burstiness import calculate_burstiness, normalize_burstiness
from app.services.repetition import calculate_repetition_score, normalize_repetition

logger = get_logger(__name__)


def calculate_final_score(text: str, sentences: List[str]) -> Dict[str, any]:
    """Calculate final AI detection score.
    
    Args:
        text: Full text
        sentences: List of sentences
        
    Returns:
        Dictionary with score, label, confidence, and metrics
    """
    # Calculate individual metrics
    perplexity = calculate_perplexity(text)
    burstiness = calculate_burstiness(sentences)
    repetition = calculate_repetition_score(text)
    
    # Calculate sentence-level perplexities for variance
    sentence_scores = calculate_sentence_perplexities(sentences)
    variance = calculate_perplexity_variance(sentence_scores)
    
    # Normalize to 0-100 scale
    perplexity_score = normalize_perplexity(perplexity)
    burstiness_score = normalize_burstiness(burstiness)
    repetition_score = normalize_repetition(repetition)
    variance_score = normalize_variance(variance)
    
    # Weighted combination
    final_score = (
        perplexity_score * settings.perplexity_weight +
        burstiness_score * settings.burstiness_weight +
        repetition_score * settings.repetition_weight +
        variance_score * settings.variance_weight
    )
    
    # Determine label and confidence
    if final_score >= settings.ai_threshold:
        label = "AI-generated"
        confidence = "high" if final_score >= 85 else "medium"
    elif final_score <= settings.human_threshold:
        label = "Human-written"
        confidence = "high" if final_score <= 15 else "medium"
    else:
        label = "Uncertain"
        confidence = "low"
    
    logger.info(
        f"Final score: {final_score:.2f} ({label}, {confidence} confidence) - "
        f"PPL={perplexity_score:.1f}, Burst={burstiness_score:.1f}, Rep={repetition_score:.1f}, Var={variance_score:.1f}"
    )
    
    return {
        'score': round(final_score, 2),
        'label': label,
        'confidence': confidence,
        'metrics': {
            'perplexity': round(perplexity, 2),
            'perplexity_score': round(perplexity_score, 2),
            'burstiness': round(burstiness, 3),
            'burstiness_score': round(burstiness_score, 2),
            'repetition': round(repetition, 3),
            'repetition_score': round(repetition_score, 2),
            'perplexity_variance': round(variance, 4),
            'perplexity_variance_score': round(variance_score, 2)
        }
    }


def calculate_sentence_scores(sentences: List[str]) -> List[Dict[str, any]]:
    """Calculate AI scores for individual sentences.
    
    Args:
        sentences: List of sentences
        
    Returns:
        List of dicts with sentence text and AI score
    """
    sentence_perplexities = calculate_sentence_perplexities(sentences)
    
    results = []
    for item in sentence_perplexities:
        # Normalize perplexity to score
        score = normalize_perplexity(item['perplexity'])
        
        results.append({
            'text': item['text'],
            'score': round(score, 2)
        })
    
    return results
