"""Pydantic schemas for API requests and responses."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator


class AnalyzeRequest(BaseModel):
    """Request schema for text analysis."""
    
    text: str = Field(..., min_length=10, max_length=10000, description="Text to analyze")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v


class SentenceScore(BaseModel):
    """Score for individual sentence."""
    
    text: str
    score: float = Field(..., ge=0, le=100)


class Metrics(BaseModel):
    """Individual metric values."""
    
    perplexity: float
    perplexity_score: float
    burstiness: float
    burstiness_score: float
    repetition: float
    repetition_score: float
    perplexity_variance: float
    perplexity_variance_score: float
    cv_score: float
    skew_score: float


class AnalyzeResponse(BaseModel):
    """Response schema for text analysis."""
    
    score: float = Field(..., ge=0, le=100, description="Overall AI detection score (0-100)")
    label: str = Field(..., description="Classification label")
    confidence: str = Field(..., description="Confidence level")
    metrics: Metrics
    is_reliable: bool
    sentence_scores: List[SentenceScore]
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 75.5,
                "label": "AI-generated",
                "confidence": "high",
                "metrics": {
                    "perplexity": 12.3,
                    "perplexity_score": 78.5,
                    "burstiness": 0.45,
                    "burstiness_score": 55.0,
                    "repetition": 0.68,
                    "repetition_score": 68.0
                },
                "sentence_scores": [
                    {"text": "First sentence.", "score": 80.2},
                    {"text": "Second sentence.", "score": 70.8}
                ]
            }
        }
