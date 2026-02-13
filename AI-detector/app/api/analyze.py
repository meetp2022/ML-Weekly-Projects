"""API endpoints for text analysis."""
from fastapi import APIRouter, HTTPException
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.preprocessing import preprocess_text
from app.services.scoring import calculate_final_score, calculate_sentence_scores
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze text for AI detection.
    
    Args:
        request: Analysis request with text
        
    Returns:
        Analysis results with score, metrics, and sentence-level scores
    """
    try:
        logger.info(f"Analyzing text ({len(request.text)} chars)")
        
        # Preprocess text
        cleaned_text, sentences = preprocess_text(request.text)
        
        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="No valid sentences found in text"
            )
        
        # Calculate scores
        result = calculate_final_score(cleaned_text, sentences)
        sentence_scores = calculate_sentence_scores(sentences, result['score'])
        
        # Build response
        response = AnalyzeResponse(
            score=result['score'],
            label=result['label'],
            confidence=result['confidence'],
            metrics=result['metrics'],
            is_reliable=result['is_reliable'],
            sentence_scores=sentence_scores
        )
        
        logger.info(f"Analysis complete: {result['label']} ({result['score']:.2f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing text: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing text: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Status message
    """
    return {"status": "healthy", "service": "ai-text-detector"}
