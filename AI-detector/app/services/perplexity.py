"""Perplexity calculation using GPT-2."""
from typing import List, Dict
import torch
import numpy as np

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
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=settings.max_length)
    
    # Move to device
    device = torch.device(settings.device)
    input_ids = encodings.input_ids.to(device)
    
    # Calculate loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()
    
    logger.debug(f"Document perplexity: {perplexity:.2f}")
    
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
