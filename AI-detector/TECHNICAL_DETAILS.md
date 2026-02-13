# AIChecking | Technical Methodology & Architecture

This document provides a technical overview of how the **AIChecking** detection engine operates, the technologies powering it, and the roadmap for improving detection accuracy.

---

## 🚀 Technology Stack

AIChecking is built using a modern, scalable AI stack:

- **Core Engine**: Python 3.9+
- **Machine Learning Framework**: [PyTorch](https://pytorch.org/) (optimized for CPU inference)
- **Transformer Models**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Primary Model**: `distilgpt2` (a distilled, high-speed version of GPT-2)
- **Natural Language Processing**: [NLTK](https://www.nltk.org/) for advanced sentence tokenization
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) for high-concurrency asynchronous processing
- **Frontend**: Vanilla JS/CSS for maximum performance and compatibility

---

## 🔍 How Detection Works

Our engine uses a **multi-dimensional statistical approach**. Instead of looking for a single "AI signature," it analyzes text across four key metrics:

### 1. Perplexity (Predictability)
We pass the text through a GPT-2 model to calculate how "surprised" the model is by the next word.
- **Low Perplexity**: The text follows the exact statistical patterns the model was trained on (Likely AI).
- **High Perplexity**: The text contains creative or non-linear word choices that a machine wouldn't typically predict (Likely Human).

### 2. Burstiness (Structural Variation)
We measure the coefficient of variation in sentence lengths.
- **Uniform Lengths**: AI tends to generate sentences of similar length and complexity (Low Burstiness).
- **Varied Lengths**: Human writers naturally mix short, punchy sentences with long, descriptive ones (High Burstiness).

### 3. Repetition Detection
We perform n-gram analysis (bigrams and trigrams) to find repeating patterns.
- **Repetitive Structures**: Machines often fall into loops of similar phrasing or structural patterns.

### 4. Stylometric Variance
We analyze how perplexity fluctuates *within* the document.
- **Consistent Predictability**: AI writing is monotonously predictable from start to finish.
- **Fluctuating Predictability**: Humans often start a thought simply and then move into complex explanations, leading to high internal variance.

---

## 📈 Improving Accuracy Significantly

The current demo-grade engine can be significantly upgraded to reach "enterprise-grade" accuracy (95%+) through the following methods:

### 1. Model Scaling
Upgrading from `distilgpt2` (82 million parameters) to larger models like **GPT-2 Large** (774M) or **Llama-3-8B** would allow for much more nuanced predictability analysis.

### 2. Ensemble Methods
Instead of relying on one model, we can use a "Council of Models." By cross-referencing results from `RoBERTa`, `BERT`, and `GPT-2`, we can identify consensus among different architectures, drastically reducing false positives.

### 3. Fine-Tuning on Contrastive Datasets
Currently, we use a general-purpose model. By fine-tuning a classifier specifically on a dataset of **50% AI-generated** and **50% Human-verified** pairs (across various domains like law, medicine, and creative writing), the model can learn subtle "synthetic fingerprints."

### 4. Stylometric Profiling
Integrating classical NLP features—such as the frequency of stop words, use of passive voice, and vocabulary richness index—adds a layer of "human-ness" detection that raw statistical models sometimes miss.

### 5. Semantic Consistency Analysis
AI models occasionally "hallucinate" or lose the logical thread over long documents. Implementing a semantic graph to check if the text maintains logical consistency from beginning to end is a powerful way to flag AI content.

---

## 💰 Cost Optimization & Scaling Strategy

Improving accuracy doesn't always require expensive enterprise GPUs. Here is the roadmap for cost-optimized growth:

### 1. Free & Low-Cost Infrastructure
- **Hugging Face Spaces**: Offers a robust, forever-free tier for CPU-based inference. Many of the "Large" models (GPT-2 Large, BERT) can run efficiently here without a dedicated GPU.
- **Quantization (INT8/FP16)**: We can reduce the memory footprint of larger models by 50-70% using quantization. This allows us to run more powerful models (like Llama-3-3B) on standard 1GB-2GB RAM instances.

### 2. Open-Source vs. Paid APIs
- **Avoiding Token Costs**: Relying on local open-source models (Hugging Face) instead of OpenAI/Perplexity APIs ensures your monthly costs remain **fixed at $0** regardless of how many users analyze text.
- **Small-but-Mighty Models**: Modern "Small Language Models" (SLMs) like **Microsoft Phi-3** or **Google Gemma-2B** offer performance comparable to much larger models but can be hosted on basic cloud servers for ~$5-10/month.

### 3. Progressive Scaling
- **Phase 1 ($0/mo)**: Stay on GitHub Pages + DigitalOcean Basic. Optimize the current `distilgpt2` with better preprocessing and stylometric rules.
- **Phase 2 (~$12/mo)**: Upgrade to a 2GB RAM instance to host an ensemble of three small models (BERT + DistilGPT-2 + RoBERTa).
- **Phase 3 (Variable)**: Only introduce GPU costs if you reach 10,000+ daily checks, as CPU inference is sufficient for lower traffic volumes.
