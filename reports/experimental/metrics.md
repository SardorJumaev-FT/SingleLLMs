# Financial Misinformation Detection - Model Evaluation Report

## Overview

This report presents the evaluation results for three single-LLM approaches to financial misinformation detection:

1. **Zero-shot Standard**: Simple task instruction + claim only
2. **Zero-shot Chain-of-Thought**: "Let's think step-by-step" reasoning
3. **Few-shot Chain-of-Thought**: 8 in-context examples with reasoning

## Performance Summary

### Model Comparison

| Model                      |   Accuracy |   Precision (True) |   Recall (True) |   F1-True |   Precision (False) |   Recall (False) |   F1-False |   Macro F1 |   N Samples |
|:---------------------------|-----------:|-------------------:|----------------:|----------:|--------------------:|-----------------:|-----------:|-----------:|------------:|
| Zero-shot Chain-of-Thought |      0.737 |              0.833 |           0.556 |     0.667 |               0.692 |              0.9 |      0.783 |      0.725 |          19 |
| Zero-shot Standard         |      0.7   |              0.833 |           0.5   |     0.625 |               0.643 |              0.9 |      0.75  |      0.688 |          20 |
| Few-shot Chain-of-Thought  |      0.65  |              0.8   |           0.4   |     0.533 |               0.6   |              0.9 |      0.72  |      0.627 |          20 |

## Detailed Results

### Zero-shot Standard

- **Sample Size**: 20
- **Accuracy**: 0.700
- **F1-True (Real Claims)**: 0.625
- **F1-False (Fake Claims)**: 0.750
- **Macro F1**: 0.688

**Confusion Matrix:**

|       | Pred False | Pred True |
|-------|------------|-----------||
| **True False** | 9 | 1 |
| **True True**  | 5 | 5 |

### Zero-shot Chain-of-Thought

- **Sample Size**: 19
- **Accuracy**: 0.737
- **F1-True (Real Claims)**: 0.667
- **F1-False (Fake Claims)**: 0.783
- **Macro F1**: 0.725

**Confusion Matrix:**

|       | Pred False | Pred True |
|-------|------------|-----------||
| **True False** | 9 | 1 |
| **True True**  | 4 | 5 |

### Few-shot Chain-of-Thought

- **Sample Size**: 20
- **Accuracy**: 0.650
- **F1-True (Real Claims)**: 0.533
- **F1-False (Fake Claims)**: 0.720
- **Macro F1**: 0.627

**Confusion Matrix:**

|       | Pred False | Pred True |
|-------|------------|-----------||
| **True False** | 9 | 1 |
| **True True**  | 6 | 4 |

## Methodology

### Dataset
- **Source**: FinFact dataset (financial fact-checking claims)
- **Total Records**: 3,369 claims
- **Binary Classification**: True/False (excluded NEI labels)
- **Final Dataset**: 2,767 claims
- **Split**: 70% train, 15% validation, 15% test

### Models
- **Base Model**: GPT-4o-mini
- **Temperature**: 0.3
- **Evaluation**: Closed-book (no internet access)

### Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-True**: F1 score for real claims
- **F1-False**: F1 score for fake claims
- **Macro F1**: Average of class-specific F1 scores

---
*Report generated automatically by the evaluation pipeline.*
