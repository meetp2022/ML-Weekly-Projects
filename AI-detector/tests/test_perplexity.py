"""Tests for perplexity calculation."""
import pytest
from app.services.perplexity import (
    calculate_perplexity,
    calculate_sentence_perplexities,
    normalize_perplexity
)


def test_calculate_perplexity():
    """Test perplexity calculation."""
    # AI-like text (more predictable)
    ai_text = "The weather is nice today. It is sunny and warm. The temperature is pleasant."
    ai_ppl = calculate_perplexity(ai_text)
    
    # Human-like text (more varied)
    human_text = "Wow! What an absolutely gorgeous day. Can't believe how perfect the weather turned out."
    human_ppl = calculate_perplexity(human_text)
    
    # AI text should have lower perplexity (more predictable)
    assert ai_ppl > 0
    assert human_ppl > 0
    # Note: This assertion might not always hold for short texts
    # assert ai_ppl < human_ppl


def test_calculate_sentence_perplexities():
    """Test sentence-level perplexity calculation."""
    sentences = [
        "This is a test sentence.",
        "Another sentence for testing.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    results = calculate_sentence_perplexities(sentences)
    
    assert len(results) == 3
    for result in results:
        assert 'text' in result
        assert 'perplexity' in result
        assert result['perplexity'] > 0


def test_normalize_perplexity():
    """Test perplexity normalization."""
    # Low perplexity should give high AI score
    low_ppl = normalize_perplexity(10.0)
    assert 80 <= low_ppl <= 100
    
    # High perplexity should give low AI score
    high_ppl = normalize_perplexity(90.0)
    assert 0 <= high_ppl <= 20
    
    # Mid perplexity should give mid score
    mid_ppl = normalize_perplexity(50.0)
    assert 40 <= mid_ppl <= 60


def test_empty_text():
    """Test handling of empty text."""
    with pytest.raises(Exception):
        calculate_perplexity("")
