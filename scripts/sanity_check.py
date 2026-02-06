"""Sanity check script to verify all components work."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.preprocessing import preprocess_text
from app.services.perplexity import calculate_perplexity
from app.services.burstiness import calculate_burstiness
from app.services.repetition import calculate_repetition_score
from app.services.scoring import calculate_final_score
from app.models.gpt2_loader import gpt2_loader


def main():
    """Run sanity checks."""
    print("=" * 60)
    print("AI Text Detector - Sanity Check")
    print("=" * 60)
    
    # Test text
    test_text = """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning models can now perform tasks that were once thought
    to be exclusively human. Natural language processing has advanced
    significantly, enabling computers to understand and generate text.
    """
    
    print("\n1. Testing text preprocessing...")
    try:
        cleaned, sentences = preprocess_text(test_text)
        print(f"   ✓ Preprocessed {len(sentences)} sentences")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n2. Testing model loading...")
    try:
        model, tokenizer = gpt2_loader.load()
        print(f"   ✓ Model loaded: {type(model).__name__}")
        print(f"   ✓ Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n3. Testing perplexity calculation...")
    try:
        perplexity = calculate_perplexity(cleaned)
        print(f"   ✓ Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n4. Testing burstiness calculation...")
    try:
        burstiness = calculate_burstiness(sentences)
        print(f"   ✓ Burstiness: {burstiness:.3f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n5. Testing repetition calculation...")
    try:
        repetition = calculate_repetition_score(cleaned)
        print(f"   ✓ Repetition: {repetition:.3f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n6. Testing final scoring...")
    try:
        result = calculate_final_score(cleaned, sentences)
        print(f"   ✓ Final Score: {result['score']:.2f}")
        print(f"   ✓ Label: {result['label']}")
        print(f"   ✓ Confidence: {result['confidence']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All sanity checks passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
