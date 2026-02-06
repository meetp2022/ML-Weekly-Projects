"""Tests for burstiness calculation."""
import pytest
from app.services.burstiness import calculate_burstiness, normalize_burstiness


def test_calculate_burstiness():
    """Test burstiness calculation."""
    # Uniform sentence lengths (low burstiness - AI-like)
    uniform_sentences = [
        "This is a test.",
        "This is another test.",
        "This is yet another test."
    ]
    uniform_burst = calculate_burstiness(uniform_sentences)
    
    # Varied sentence lengths (high burstiness - human-like)
    varied_sentences = [
        "Short.",
        "This is a medium length sentence with some words.",
        "Here we have an extremely long sentence that goes on and on with many words and clauses to demonstrate variation."
    ]
    varied_burst = calculate_burstiness(varied_sentences)
    
    # Varied should have higher burstiness
    assert 0 <= uniform_burst <= 1
    assert 0 <= varied_burst <= 1
    assert varied_burst > uniform_burst


def test_normalize_burstiness():
    """Test burstiness normalization."""
    # High burstiness (human-like) should give low AI score
    high_burst = normalize_burstiness(0.8)
    assert 0 <= high_burst <= 30
    
    # Low burstiness (AI-like) should give high AI score
    low_burst = normalize_burstiness(0.2)
    assert 70 <= low_burst <= 100


def test_single_sentence():
    """Test handling of single sentence."""
    sentences = ["This is a single sentence."]
    burstiness = calculate_burstiness(sentences)
    
    # Should return neutral score
    assert burstiness == 0.5


def test_empty_sentences():
    """Test handling of empty sentences."""
    burstiness = calculate_burstiness([])
    assert burstiness == 0.5
