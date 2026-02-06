"""Tests for scoring logic."""
import pytest
from app.services.scoring import calculate_final_score, calculate_sentence_scores


def test_calculate_final_score():
    """Test final score calculation."""
    # AI-like text
    ai_text = "The weather is nice. It is sunny. The temperature is warm. It is pleasant."
    ai_sentences = [
        "The weather is nice.",
        "It is sunny.",
        "The temperature is warm.",
        "It is pleasant."
    ]
    
    ai_result = calculate_final_score(ai_text, ai_sentences)
    
    assert 'score' in ai_result
    assert 'label' in ai_result
    assert 'confidence' in ai_result
    assert 'metrics' in ai_result
    
    assert 0 <= ai_result['score'] <= 100
    assert ai_result['label'] in ['AI-generated', 'Human-written', 'Uncertain']
    assert ai_result['confidence'] in ['high', 'medium', 'low']


def test_calculate_sentence_scores():
    """Test sentence-level scoring."""
    sentences = [
        "This is a test sentence.",
        "Another sentence for testing.",
        "The quick brown fox jumps."
    ]
    
    results = calculate_sentence_scores(sentences)
    
    assert len(results) == 3
    for result in results:
        assert 'text' in result
        assert 'score' in result
        assert 0 <= result['score'] <= 100


def test_metrics_structure():
    """Test that metrics have correct structure."""
    text = "This is a test. It has multiple sentences. They are for testing."
    sentences = ["This is a test.", "It has multiple sentences.", "They are for testing."]
    
    result = calculate_final_score(text, sentences)
    metrics = result['metrics']
    
    assert 'perplexity' in metrics
    assert 'perplexity_score' in metrics
    assert 'burstiness' in metrics
    assert 'burstiness_score' in metrics
    assert 'repetition' in metrics
    assert 'repetition_score' in metrics
    
    # All scores should be 0-100
    assert 0 <= metrics['perplexity_score'] <= 100
    assert 0 <= metrics['burstiness_score'] <= 100
    assert 0 <= metrics['repetition_score'] <= 100
