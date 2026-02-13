"""RoBERTa classifier loader with singleton pattern."""
from typing import Optional, Tuple
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DetectorLoader:
    """Singleton class for loading and caching the specialized RoBERTa detector."""
    
    _instance: Optional['DetectorLoader'] = None
    _model: Optional[RobertaForSequenceClassification] = None
    _tokenizer: Optional[RobertaTokenizer] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self) -> Tuple[RobertaForSequenceClassification, RobertaTokenizer]:
        """Load RoBERTa model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self._model is None or self._tokenizer is None:
            logger.info(f"Loading classifier: {settings.classifier_model_name}")
            
            self._tokenizer = RobertaTokenizer.from_pretrained(settings.classifier_model_name)
            self._model = RobertaForSequenceClassification.from_pretrained(
                settings.classifier_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            
            # Set to evaluation mode
            self._model.eval()
            
            # Move to device
            device = torch.device(settings.device)
            self._model.to(device)
            
            # Optional: Disable gradients globally to save memory
            torch.set_grad_enabled(False)
            
            logger.info(f"Classifier loaded successfully on {settings.device}")
        
        return self._model, self._tokenizer
    
    @property
    def model(self) -> RobertaForSequenceClassification:
        """Get the model instance."""
        if self._model is None:
            self.load()
        return self._model
    
    @property
    def tokenizer(self) -> RobertaTokenizer:
        """Get the tokenizer instance."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer


# Global instance
detector_loader = DetectorLoader()
