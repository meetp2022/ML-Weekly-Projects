FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies using CPU-only Torch to save memory
RUN pip install --no-cache-dir torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Pre-download DistilGPT-2 and RoBERTa Detector to cache in image layers
RUN python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2Tokenizer.from_pretrained('distilgpt2'); GPT2LMHeadModel.from_pretrained('distilgpt2')"
RUN python -c "from transformers import RobertaForSequenceClassification, RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base-openai-detector'); RobertaForSequenceClassification.from_pretrained('roberta-base-openai-detector')"

# Copy application code
COPY . .

# Expose port (Documentation only, cloud providers will override)
EXPOSE 8000

# Run the application with dynamic port for cloud compatibility
# Fallback to 8000 if PORT is not set
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
