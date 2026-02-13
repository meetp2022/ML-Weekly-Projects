"""Perplexity calculation using GPT-2."""
from typing import List, Dict
import torch
import numpy as np
from scipy.stats import skew

from app.models.gpt2_loader import gpt2_loader
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def calculate_perplexity(text: str) -> float:
    """Calculate perplexity for entire text.
    
    Args:
        text: Input text
        
    Returns:
        Perplexity score (lower = more AI-like)
    """
    model, tokenizer = gpt2_loader.load()
    
    # Tokenize
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=settings.max_token_length)
    
    # Move to device
    device = torch.device(settings.device)
    input_ids = encodings.input_ids.to(device)
    
    # Calculate loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()
    
    # NEW: Length Protection Logic
    # For very short texts (under 150 words), predictability is NOT a bug, it's a feature of simple English.
    # We apply a "Benefit of the Doubt" multiplier that RAISES perplexity for short samples.
    word_count = len(text.split())
    if word_count < 150:
        # Scale up: a 10-word text gets a ~2x boost to perplexity to avoid false positives
        protection_factor = 2.0 - min(word_count / 150.0, 1.0)
        perplexity = perplexity * protection_factor
    elif word_count > 500:
        # For long texts, we apply a mild creative penalty (AI stays too predictable over time)
        length_penalty = np.log10(word_count) / 2.5
        perplexity = perplexity * length_penalty
    
    logger.debug(f"Document perplexity (calibrated): {perplexity:.2f}")
    
    return perplexity


def calculate_sentence_perplexities(sentences: List[str]) -> List[Dict[str, any]]:
    """Calculate perplexity for each sentence.
    
    Args:
        sentences: List of sentences
        
    Returns:
        List of dicts with sentence text and perplexity score
    """
    model, tokenizer = gpt2_loader.load()
    device = torch.device(settings.device)
    
    results = []
    
    for sentence in sentences:
        try:
            # Tokenize
            encodings = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(device)
            
            # Skip very short sentences
            if input_ids.shape[1] < 3:
                continue
            
            # Calculate loss
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            
            perplexity = torch.exp(loss).item()
            
            results.append({
                'text': sentence,
                'perplexity': perplexity
            })
            
        except Exception as e:
            logger.warning(f"Error calculating perplexity for sentence: {e}")
            continue
    
    return results


def normalize_perplexity(perplexity: float, min_ppl: float = 10.0, max_ppl: float = 300.0) -> float:
    """Normalize perplexity with calibrated DistilGPT-2 ranges.
    
    Min PPL 10: Typical for very simple AI or extremely simple repetitive human text.
    Max PPL 300: Typical for complex human academic writing.
    """
    # Clamp to range
    perplexity = np.clip(perplexity, min_ppl, max_ppl)
    
    # Invert: lower perplexity = higher AI score
    normalized = 100 * (1 - (perplexity - min_ppl) / (max_ppl - min_ppl))
    
    return float(normalized)


def calculate_perplexity_variance(sentence_scores: List[Dict[str, any]]) -> float:
    """Calculate the variation in perplexity across sentences.
    
    Human writing usually has higher variance (changes in tone/complexity).
    AI writing usually has lower variance (very stable predictability).
    
    Args:
        sentence_scores: List of dicts with 'perplexity' key
        
    Returns:
        Variance (standard deviation) of perplexities
    """
    if len(sentence_scores) < 3:
        return 0.0
    
    perplexities = [s['perplexity'] for s in sentence_scores]
    
    # Calculate standard deviation
    std_dev = np.std(perplexities)
    
    logger.debug(f"Perplexity variance (std_dev): {std_dev:.2f}")
    
    return float(std_dev)


def normalize_variance(variance: float, threshold: float = 15.0) -> float:
    """Normalize perplexity variance to 0-100 score."""
    normalized = 100 * (1 - min(variance / threshold, 1.0))
    return float(normalized)


def calculate_perplexity_distribution(sentence_scores: List[Dict[str, any]]) -> Dict[str, float]:
    """Calculate advanced distribution metrics for sentence perplexities."""
    if not sentence_scores:
        return {'std': 0.0, 'cv': 0.0, 'skew': 0.0}
    
    ppls = [s['perplexity'] for s in sentence_scores]
    mean = np.mean(ppls)
    std = np.std(ppls)
    cv = std / mean if mean > 0 else 0
    
    # Skewness: AI writing is usually very symmetric (predictable), 
    # human writing has long tails of high perplexing words.
    text_skew = skew(ppls) if len(ppls) > 2 else 0.0
    
    return {
        'std': float(std),
        'cv': float(cv),
        'skew': float(text_skew)
    }
