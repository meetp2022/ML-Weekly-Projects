"""GPT-2 model loader with singleton pattern."""
from typing import Optional, Tuple
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GPT2Loader:
    """Singleton class for loading and caching GPT-2 model."""
    
    _instance: Optional['GPT2Loader'] = None
    _model: Optional[GPT2LMHeadModel] = None
    _tokenizer: Optional[GPT2Tokenizer] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
        """Load GPT-2 model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self._model is None or self._tokenizer is None:
            logger.info(f"Loading model: {settings.model_name}")
            
            self._tokenizer = GPT2Tokenizer.from_pretrained(settings.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(
                settings.model_name,
                low_cpu_mem_usage=True,  # Reduces RAM spike during loading
                torch_dtype=torch.float32 # Ensure standard precision for stability
            )
            
            # Set to evaluation mode
            self._model.eval()
            
            # Move to device
            device = torch.device(settings.device)
            self._model.to(device)
            
            # Optional: Disable gradients globally to save memory
            torch.set_grad_enabled(False)
            
            logger.info(f"Model loaded successfully on {settings.device}")
        
        return self._model, self._tokenizer
    
    @property
    def model(self) -> GPT2LMHeadModel:
        """Get the model instance."""
        if self._model is None:
            self.load()
        return self._model
    
    @property
    def tokenizer(self) -> GPT2Tokenizer:
        """Get the tokenizer instance."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer


# Global instance
gpt2_loader = GPT2Loader()
