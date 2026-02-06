"""Benchmark script with example AI and human texts."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.preprocessing import preprocess_text
from app.services.scoring import calculate_final_score


# Example texts
AI_EXAMPLES = [
    """
    Artificial intelligence is a rapidly evolving field that encompasses various 
    technologies and methodologies. Machine learning, a subset of AI, enables 
    computers to learn from data without explicit programming. Deep learning, 
    which uses neural networks, has shown remarkable success in image recognition 
    and natural language processing tasks.
    """,
    """
    Climate change is one of the most pressing issues facing humanity today. 
    Rising global temperatures are causing ice caps to melt and sea levels to rise. 
    Extreme weather events are becoming more frequent and severe. Scientists agree 
    that immediate action is needed to reduce greenhouse gas emissions and mitigate 
    the effects of climate change.
    """
]

HUMAN_EXAMPLES = [
    """
    Honestly? I can't believe how much AI has changed in just a few years. 
    Remember when chatbots were basically useless? Now they're... well, still 
    sometimes useless, but in much more sophisticated ways! The tech is wild, 
    though - makes you wonder what's next.
    """,
    """
    So I was reading about climate stuff yesterday, and wow, it's actually pretty 
    scary. Like, we all know it's bad, but seeing the actual numbers? Yikes. 
    My cousin works in environmental science and she says even the scientists are 
    more worried than they let on publicly. Not great, folks.
    """
]


def analyze_example(text, label):
    """Analyze a single example."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {label}")
    print(f"{'=' * 60}")
    print(f"Text preview: {text[:100]}...")
    
    cleaned, sentences = preprocess_text(text)
    result = calculate_final_score(cleaned, sentences)
    
    print(f"\nResults:")
    print(f"  Score: {result['score']:.2f} / 100")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"\nMetrics:")
    print(f"  Perplexity: {result['metrics']['perplexity']:.2f} (score: {result['metrics']['perplexity_score']:.1f})")
    print(f"  Burstiness: {result['metrics']['burstiness']:.3f} (score: {result['metrics']['burstiness_score']:.1f})")
    print(f"  Repetition: {result['metrics']['repetition']:.3f} (score: {result['metrics']['repetition_score']:.1f})")
    
    return result


def main():
    """Run benchmark on example texts."""
    print("=" * 60)
    print("AI Text Detector - Benchmark Examples")
    print("=" * 60)
    
    ai_scores = []
    human_scores = []
    
    # Analyze AI examples
    print("\n\n### AI-GENERATED EXAMPLES ###")
    for i, text in enumerate(AI_EXAMPLES, 1):
        result = analyze_example(text, f"AI Example {i}")
        ai_scores.append(result['score'])
    
    # Analyze human examples
    print("\n\n### HUMAN-WRITTEN EXAMPLES ###")
    for i, text in enumerate(HUMAN_EXAMPLES, 1):
        result = analyze_example(text, f"Human Example {i}")
        human_scores.append(result['score'])
    
    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAI Examples:")
    print(f"  Average Score: {sum(ai_scores) / len(ai_scores):.2f}")
    print(f"  Range: {min(ai_scores):.2f} - {max(ai_scores):.2f}")
    
    print(f"\nHuman Examples:")
    print(f"  Average Score: {sum(human_scores) / len(human_scores):.2f}")
    print(f"  Range: {min(human_scores):.2f} - {max(human_scores):.2f}")
    
    print(f"\nSeparation: {abs(sum(ai_scores) / len(ai_scores) - sum(human_scores) / len(human_scores)):.2f} points")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
