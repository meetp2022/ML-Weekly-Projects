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
from app.services.modality import detect_modality
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
    
    # MODALITY DETECTION
    modality_info = detect_modality(text)
    is_technical = modality_info['type'] == "TECHNICAL"
    
    # Normalize to 0-100 scale
    perplexity_score = normalize_perplexity(perplexity)
    burstiness_score = normalize_burstiness(burstiness)
    repetition_score = normalize_repetition(repetition)
    variance_score = normalize_variance(variance)
    
    # Aggregated Sentence Features (The Core of Consensus)
    # 1. AI Sentence Ratio (% of sentences > 65%)
    # 2. Mean Sentence Probability
    # 3. Longest AI Streak
    local_scores = [normalize_perplexity(s['perplexity']) for s in sentence_scores]
    ai_sentence_count = sum(1 for s in local_scores if s >= 65)
    ai_ratio = (ai_sentence_count / len(sentences)) * 100 if sentences else 0
    mean_prob = np.mean(local_scores) if local_scores else 0
    
    # Calculate Streak
    max_streak = 0
    current_streak = 0
    for s in local_scores:
        if s >= 65:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    streak_bonus = min(max_streak / 5.0, 1.0) * 15.0 # Max 15 points bonus for long streaks
    
    # New Scientific Scores (Inverted for AI risk)
    cv_score = 100 * (1 - min(dist_metrics['cv'] / 1.5, 1.0))
    skew_score = 100 * (1 - min(max(dist_metrics['skew'], 0) / 4.0, 1.0))
    
    # AGGREGATED CONSENSUS SCORE
    # We combine the Document-Level Classifier (RoBERTa) with the Sentence-Level Aggregates
    statistical_base = (
        perplexity_score * 0.30 +
        ai_ratio * 0.30 +
        mean_prob * 0.20 +
        repetition_score * 0.10 +
        cv_score * 0.05 +
        skew_score * 0.05
    )
    
    # Add streak bonus for consistent patterns
    statistical_base = min(statistical_base + streak_bonus, 100)
    
    # Final Hybrid Score (Classifier + Aggregated Stats)
    if is_technical:
        # For technical text, we trust the statistical patterns more than the prose-trained classifier
        final_score = statistical_base
        modality_warning = "Technical/Code detected - reliability is reduced for this modality."
    else:
        final_score = (classifier_ai_prob * 0.50) + (statistical_base * 0.50)
        modality_warning = None
    
    # Refined Thresholds
    if final_score >= 65:
        label = "AI-generated"
        confidence = "high" if final_score >= 85 else "medium"
    elif final_score <= 35:
        label = "Human-written"
        confidence = "high" if final_score <= 15 else "medium"
    else:
        label = "Uncertain"
        confidence = "low"
    
    # Reliability check
    is_reliable = len(text) >= 150 and not is_technical
    
    logger.info(
        f"Final score: {final_score:.2f} ({label}) - Modality: {modality_info['type']} - "
        f"AI Ratio={ai_ratio:.1f}%, Mean Prob={mean_prob:.1f}%, PPL={perplexity_score:.1f}"
    )
    
    return {
        'score': round(final_score, 2),
        'label': label,
        'confidence': confidence,
        'is_reliable': is_reliable,
        'modality': modality_info['type'],
        'modality_warning': modality_warning,
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
