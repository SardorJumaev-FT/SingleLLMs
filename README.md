# Financial Misinformation Detection — Single‑LLM Baseline Study

A comprehensive implementation of three single-LLM approaches for financial misinformation detection using the FinFact dataset.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

This project implements and compares three independent OpenAI GPT-4o-mini-based classifiers for detecting financial misinformation:

1. **Zero-shot Standard** — Simple task instruction + claim only
2. **Zero-shot Chain-of-Thought (CoT)** — "Let's think step-by-step" reasoning
3. **Few-shot Chain-of-Thought** — 8 in-context examples with reasoning

### Key Features

- ✅ **No fine-tuning** — Pure prompt engineering approach
- ✅ **Closed-book evaluation** — No internet access during classification
- ✅ **Comprehensive evaluation** — Accuracy, precision, recall, F1 scores
- ✅ **Reproducible results** — Deterministic data splits and random seeds
- ✅ **Production-ready** — Error handling, logging, and robust parsing

## 📊 Dataset

- **Source**: FinFact dataset (financial fact-checking claims)
- **Total Records**: 3,369 claims
- **Binary Classification**: True/False (excludes NEI labels)
- **Final Dataset**: 2,767 claims
- **Split**: 70% train, 15% validation, 15% test

## 🏗️ Project Structure

```
SingleLLMs/
├── data/                       # Data files and splits
│   ├── finfact.json           # Original dataset
│   ├── train.csv              # Training split
│   ├── val.csv                # Validation split
│   ├── test.csv               # Test split
│   └── few_shot_examples.json # Selected examples for few-shot
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis (Google Colab)
├── prompts/                   # Prompt templates
│   ├── standard.txt           # Zero-shot standard template
│   ├── cot_zero.txt          # Zero-shot CoT template
│   └── cot_fewshot.txt       # Few-shot CoT template
├── scripts/                   # Main implementation
│   ├── data_pipeline.py       # Data loading and preprocessing
│   ├── zeroshot_standard.py   # Model A: Zero-shot standard
│   ├── zeroshot_cot.py       # Model B: Zero-shot CoT
│   ├── fewshot_cot.py        # Model C: Few-shot CoT
│   └── evaluate.py           # Evaluation pipeline
├── reports/                   # Generated reports
│   ├── metrics.md            # Evaluation report
│   ├── model_comparison.csv  # Comparison table
│   └── detailed_results.json # Detailed metrics
├── tests/
│   └── test_pipeline.py      # Unit tests
├── requirements.txt          # Dependencies
├── PLANNING.md              # Project planning document
├── TASK.md                  # Task tracking
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- ~500MB disk space for dataset

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd SingleLLMs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run data pipeline:**
   ```bash
   python scripts/data_pipeline.py
   ```

### Basic Usage

Run individual models on the test set:

```bash
# Zero-shot Standard
python scripts/zeroshot_standard.py --model gpt-4o-mini --temp 0.3

# Zero-shot Chain-of-Thought
python scripts/zeroshot_cot.py --model gpt-4o-mini --temp 0.3

# Few-shot Chain-of-Thought
python scripts/fewshot_cot.py --model gpt-4o-mini --temp 0.3
```

Evaluate all models and generate reports:

```bash
python scripts/evaluate.py --standard preds_standard.csv --cot preds_cot.csv --fewshot preds_fewshot.csv
```

## 📈 Detailed Usage

### Data Pipeline

The data pipeline handles loading, preprocessing, and splitting:

```python
from scripts.data_pipeline import load_dataset, preprocess_for_binary_classification, create_splits

# Load and preprocess
df = load_dataset("finfact.json")
binary_df = preprocess_for_binary_classification(df)
train, val, test = create_splits(binary_df)
```

### Model Configuration

All models support the same command-line arguments:

```bash
python scripts/[model_script].py \
    --model gpt-4o-mini \          # OpenAI model
    --temp 0.3 \                   # Temperature (0.0-1.0)
    --input data/test.csv \        # Input dataset
    --output predictions.csv \     # Output file
    --delay 1.0                    # API rate limiting delay
```

### Evaluation Pipeline

The evaluation script computes comprehensive metrics:

```python
from scripts.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(predictions_df, "Model Name")

# Metrics include:
# - Accuracy, Precision, Recall
# - F1-True, F1-False, Macro F1
# - Confusion Matrix
# - Classification Report
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/test_pipeline.py

# Run specific test class
python -m unittest tests.test_pipeline.TestDataPipeline -v
```

Tests cover:
- Data pipeline functionality
- Model evaluation metrics
- Response parsing
- End-to-end integration

## 📊 Exploratory Data Analysis

The project includes a comprehensive EDA notebook optimized for Google Colab:

1. **Open in Colab**: Use the badge in `notebooks/eda.ipynb`
2. **Upload dataset**: Upload `finfact.json` to your Colab environment
3. **Run analysis**: Execute all cells for complete analysis

The EDA covers:
- Dataset overview and quality assessment
- Label distribution analysis
- Text characteristics and readability
- Financial terms analysis
- Temporal trends
- Author and source analysis

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_api_key_here    # Required: OpenAI API access
```

### Model Parameters

```python
# Default configuration
MODEL = "gpt-4o-mini"               # Primary model
FALLBACK_MODEL = "gpt-4"            # If primary unavailable
TEMPERATURE = 0.3                   # Consistent sampling
MAX_TOKENS = 500                    # Response length limit
TIMEOUT = 60                        # API timeout (seconds)
```

### Prompt Templates

Templates are stored in `prompts/` and use Python string formatting:

```python
# Example: prompts/standard.txt
"You are a financial fact-checking assistant. Your task is to determine whether a given financial claim is true or false based on your knowledge.\n\nRespond with exactly one word: \"True\" or \"False\"\n\nClaim: {claim}\n\nResponse:"
```

## 📋 API Usage Guidelines

### Rate Limiting

- Default delay: 1 second between API calls
- Exponential backoff on failures
- Configurable via `--delay` parameter

### Error Handling

- Automatic retries (max 3 attempts)
- Graceful degradation on API failures
- Comprehensive logging

### Cost Management

- Token usage tracking
- Estimated cost monitoring
- Batch processing support

## 🔍 Results and Reporting

### Generated Reports

1. **`reports/metrics.md`** — Comprehensive evaluation report
2. **`reports/model_comparison.csv`** — Performance comparison table
3. **`reports/detailed_results.json`** — Raw evaluation data

### Key Metrics

- **Accuracy**: Overall classification accuracy
- **F1-True**: F1 score for real financial claims
- **F1-False**: F1 score for fake financial claims
- **Macro F1**: Average of class-specific F1 scores
- **Confusion Matrix**: Detailed classification breakdown

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `python tests/test_pipeline.py`
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Create Pull Request

### Code Style

- Follow PEP8 guidelines
- Use `black` for code formatting
- Include docstrings for all functions
- Add type hints where applicable

## 📚 References

- **Chain-of-Thought Prompting**: Wei et al. (2022) — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- **FinFact Dataset**: Aman Rangapur et al. (2023) — "FinFact: A Benchmark Dataset for Multimodal Financial Fact Checking"
- **OpenAI API**: [OpenAI Platform Documentation](https://platform.openai.com/docs)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For questions, issues, or contributions:

1. **Issues**: Use GitHub Issues for bug reports
2. **Discussions**: Use GitHub Discussions for questions
3. **Email**: Contact the maintainers directly

---

**Note**: This implementation is designed for research and educational purposes. Ensure compliance with OpenAI's usage policies and your institution's ethical guidelines when using this code. 