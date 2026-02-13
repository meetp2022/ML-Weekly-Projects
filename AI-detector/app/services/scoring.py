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
    normalize_variance,
    calculate_perplexity_distribution
)
from app.services.burstiness import calculate_burstiness, normalize_burstiness
from app.services.repetition import calculate_repetition_score, normalize_repetition
from app.services.preprocessing import extract_stylometric_features

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
    
    # Calculate sentence-level perplexities and distribution
    sentence_scores = calculate_sentence_perplexities(sentences)
    dist_metrics = calculate_perplexity_distribution(sentence_scores)
    variance = dist_metrics['std']
    
    # Stylometric markers
    sty_metrics = extract_stylometric_features(text, sentences)
    
    # Normalize to 0-100 scale
    perplexity_score = normalize_perplexity(perplexity)
    burstiness_score = normalize_burstiness(burstiness)
    repetition_score = normalize_repetition(repetition)
    variance_score = normalize_variance(variance)
    
    # New Scientific Scores (Inverted for AI risk)
    # Human text has HIGH CV and HIGH Skew (long tails of complex words)
    cv_score = 100 * (1 - min(dist_metrics['cv'] / 1.5, 1.0))
    skew_score = 100 * (1 - min(max(dist_metrics['skew'], 0) / 4.0, 1.0))
    
    # Stylometric Scores
    # Human text has HIGH sentence length variance and HIGH lexical diversity
    sty_var_score = 100 * (1 - min(sty_metrics['sentence_length_var'] / 150, 1.0))
    lex_score = 100 * (1 - min(sty_metrics['lexical_diversity'] * 1.5, 1.0))
    
    # Weighted combination (Balanced to avoid False Positives)
    # Reducing Perplexity weight slightly and increasing Stylometrics/Variety
    final_score = (
        perplexity_score * 0.30 +
        burstiness_score * 0.20 +
        repetition_score * 0.10 +
        variance_score * 0.15 +
        cv_score * 0.10 +
        skew_score * 0.05 +
        sty_var_score * 0.05 +
        lex_score * 0.05
    )
    
    # Apply "Structural Variety" credit
    # If a text has high variance or burstiness, it's very likely human regardless of perplexity.
    if variance > 20 or burstiness > 0.4:
        final_score *= 0.8  # 20% Reduction in AI probability for high-variety writers
    
    # Refined Thresholds (0-35 Human | 35-65 Mixed | 65-100 AI)
    if final_score >= 65:
        label = "AI-generated"
        confidence = "high" if final_score >= 85 else "medium"
    elif final_score <= 35:
        label = "Human-written"
        confidence = "high" if final_score <= 15 else "medium"
    else:
        label = "Uncertain"
        confidence = "low"
    
    # Reliability check (Character count < 150)
    # User requested character-based threshold: "It should not reflect if the character count exceeds 150"
    is_reliable = len(text) >= 150
    
    logger.info(
        f"Final score: {final_score:.2f} ({label}) - Reliable: {is_reliable} - "
        f"PPL={perplexity_score:.1f}, Burst={burstiness_score:.1f}, CV={cv_score:.1f}, Skew={skew_score:.1f}"
    )
    
    return {
        'score': round(final_score, 2),
        'label': label,
        'confidence': confidence,
        'is_reliable': is_reliable,
        'metrics': {
            'perplexity': round(perplexity, 2),
            'perplexity_score': round(perplexity_score, 2),
            'burstiness': round(burstiness, 3),
            'burstiness_score': round(burstiness_score, 2),
            'repetition': round(repetition, 3),
            'repetition_score': round(repetition_score, 2),
            'perplexity_variance': round(variance, 4),
            'perplexity_variance_score': round(variance_score, 2),
            'cv_score': round(cv_score, 2),
            'skew_score': round(skew_score, 2)
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
