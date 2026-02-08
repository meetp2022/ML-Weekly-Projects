"""Core configuration management."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "AI Text Detector"
    environment: str = "development"
    debug: bool = True
    
    # Model settings
    model_name: str = "distilgpt2"
    max_token_length: int = 1024
    device: str = "cpu"  # Use "cuda" if GPU available
    
    # Scoring thresholds
    ai_threshold: float = 70.0
    human_threshold: float = 30.0
    
    # Weights for final score (must sum to 1.0)
    perplexity_weight: float = 0.4
    burstiness_weight: float = 0.2
    repetition_weight: float = 0.2
    variance_weight: float = 0.2
    
    # API settings
    cors_origins: list = ["*"]
    max_text_length: int = 10000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
