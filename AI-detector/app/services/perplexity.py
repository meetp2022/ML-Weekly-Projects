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
    
    # Normalize by log-length scaling to stabilize results across sizes
    # Humans tend to get more creative in longer texts, whereas AI stays predictable.
    word_count = len(text.split())
    if word_count > 0:
        length_penalty = np.log10(max(word_count, 10)) / 2.0
        perplexity = perplexity * length_penalty
    
    logger.debug(f"Document perplexity (normalized): {perplexity:.2f}")
    
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


def normalize_perplexity(perplexity: float, min_ppl: float = 5.0, max_ppl: float = 100.0) -> float:
    """Normalize perplexity to 0-100 scale (inverted: lower perplexity = higher score).
    
    Args:
        perplexity: Raw perplexity value
        min_ppl: Minimum expected perplexity
        max_ppl: Maximum expected perplexity
        
    Returns:
        Normalized score (0-100, higher = more AI-like)
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
