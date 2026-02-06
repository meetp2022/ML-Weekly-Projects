# AI Text Detector

A demo-grade AI text detector that uses perplexity, burstiness, and repetition metrics to distinguish AI-generated text from human-written content.

## Features

- 🧠 **Perplexity Analysis**: Uses GPT-2 to measure text predictability
- 📊 **Burstiness Metrics**: Analyzes sentence length variation
- 🔄 **Repetition Detection**: Identifies repetitive patterns
- 🎯 **Sentence Highlighting**: Visual feedback on AI-likely sentences
- ⚡ **Fast API**: RESTful backend with FastAPI
- 🎨 **Modern UI**: Clean, responsive web interface

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download NLTK data**:
```bash
python -c "import nltk; nltk.download('punkt')"
```

3. **Run the application**:
```bash
uvicorn app.main:app --reload
```

4. **Open your browser**:
Navigate to `http://localhost:8000`

### Docker

```bash
docker-compose up --build
```

## Project Structure

```
ai-text-detector/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/                 # API endpoints
│   ├── core/                # Configuration & logging
│   ├── services/            # Analysis services
│   ├── models/              # Model loading
│   └── schemas/             # Pydantic schemas
├── frontend/                # Web UI
├── tests/                   # Unit tests
└── scripts/                 # Utility scripts
```

## How It Works

### 1. Perplexity
Measures how "surprised" a language model is by the text. AI-generated text typically has lower perplexity because it's more predictable.

### 2. Burstiness
Analyzes variation in sentence lengths. Human writing tends to have more varied sentence structures.

### 3. Repetition
Detects repetitive n-grams and patterns. AI text often exhibits more repetition.

### 4. Scoring
Combines all metrics into a 0-100 score:
- **0-30**: Likely human-written
- **30-70**: Uncertain
- **70-100**: Likely AI-generated

## API Usage

### Analyze Text

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

**Response**:
```json
{
  "score": 75.5,
  "label": "AI-generated",
  "confidence": "high",
  "metrics": {
    "perplexity": 12.3,
    "burstiness": 0.45,
    "repetition": 0.68
  },
  "sentence_scores": [
    {"text": "First sentence.", "score": 80.2},
    {"text": "Second sentence.", "score": 70.8}
  ]
}
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run sanity check
python scripts/sanity_check.py

# Run benchmarks
python scripts/benchmark_examples.py
```

## Deployment

### Free Tier Options

1. **Render**: Deploy directly from GitHub
2. **Railway**: One-click deployment
3. **Fly.io**: Free tier with 3 VMs

### Environment Variables

- `ENVIRONMENT`: `development` or `production`
- `MODEL_NAME`: HuggingFace model name (default: `distilgpt2`)
- `MAX_LENGTH`: Maximum text length (default: `5000`)

## Limitations

- **Demo-grade**: Not production-ready for high-stakes detection
- **Model**: Uses small GPT-2 model for cost efficiency
- **Accuracy**: ~70-80% on typical cases, can be fooled
- **Language**: Optimized for English text

## Upgrade Path

To improve accuracy:
1. Use larger models (GPT-2 Large, GPT-Neo)
2. Add fine-tuned classifiers
3. Implement ensemble methods
4. Add more sophisticated features
5. Train on domain-specific data

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.
