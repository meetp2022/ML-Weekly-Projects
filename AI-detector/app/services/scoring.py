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
from app.models.detector_loader import detector_loader
import torch

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
    
    # CALCULATE CLASSIFIER SCORE (AI Fingerprints)
    # Using the specialized RoBERTa detector
    model, tokenizer = detector_loader.load()
    device = torch.device(settings.device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        # roberta-base-openai-detector labels: [Fake, Real]
        # We want the 'Fake' (AI) probability
        classifier_ai_prob = probs[0][0].item() * 100
    
    logger.debug(f"Classifier AI Probability: {classifier_ai_prob:.2f}%")
    
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
    
    # Weighted combination (Scientific Ensemble v2)
    # Increasing repetition weight and stabilizing statistical influence
    statistical_score = (
        perplexity_score * 0.40 +
        burstiness_score * 0.15 +
        repetition_score * 0.20 +
        variance_score * 0.10 +
        cv_score * 0.10 +
        skew_score * 0.05
    )
    
    # Final Hybrid Score
    final_score = (classifier_ai_prob * 0.60) + (statistical_score * 0.40)
    
    # NEW: Statistical Floor (Prevents total false negatives)
    # If the math screams AI (PPL > 90 or Repetition > 70), 
    # don't let the classifier pull it below "Uncertain" (40%)
    if (perplexity_score > 90 or repetition_score > 70) and final_score < 40:
        logger.info("Statistical Floor triggered: Overriding classifier human bias")
        final_score = 40.0 + (final_score * 0.2) # Soft floor at 40%+
    
    # Apply "Structural Variety" credit ONLY ifRoBERTa isn't highly suspicious
    if (variance > 20 or burstiness > 0.4) and classifier_ai_prob < 65:
        final_score *= 0.85
    
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


def calculate_sentence_scores(sentences: List[str], global_risk: float) -> List[Dict[str, any]]:
    """Calculate AI scores for individual sentences, synchronized with global results.
    
    Args:
        sentences: List of sentences
        global_risk: The overall document AI score
        
    Returns:
        List of dicts with sentence text and AI score
    """
    sentence_perplexities = calculate_sentence_perplexities(sentences)
    
    results = []
    for item in sentence_perplexities:
        # Blended Logic: 70% Sentence local predictability, 30% Global Signal
        # This ensures red highlights in a green document are rare but accurate.
        local_score = normalize_perplexity(item['perplexity'])
        blended_score = (local_score * 0.7) + (global_risk * 0.3)
        
        results.append({
            'text': item['text'],
            'score': round(blended_score, 2)
        })
    
    return results
