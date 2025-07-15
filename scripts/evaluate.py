#!/usr/bin/env python3
"""
Evaluation pipeline for Financial Misinformation Detection models.
Computes accuracy, precision, recall, F1-real, F1-fake metrics.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for Financial Misinformation Detection models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def load_predictions(self, file_path: str, model_name: str) -> pd.DataFrame:
        """
        Load model predictions from CSV file.
        
        Args:
            file_path (str): Path to predictions CSV.
            model_name (str): Name of the model.
            
        Returns:
            pd.DataFrame: Loaded predictions with valid entries only.
        """
        if not Path(file_path).exists():
            logger.warning(f"Predictions file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} predictions for {model_name}")
        
        # Filter out invalid predictions
        valid_mask = df['prediction'].notna() & df['true_label'].notna()
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) < len(df):
            logger.warning(f"{model_name}: {len(df) - len(valid_df)} invalid predictions filtered out")
        
        logger.info(f"{model_name}: {len(valid_df)} valid predictions for evaluation")
        return valid_df
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            
        Returns:
            Dict[str, float]: Computed metrics.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.warning("Empty predictions, returning zero metrics")
            return {
                'accuracy': 0.0,
                'precision_true': 0.0,
                'recall_true': 0.0,
                'f1_true': 0.0,
                'precision_false': 0.0,
                'recall_false': 0.0,
                'f1_false': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'n_samples': 0
            }
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics (binary classification)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Ensure we have metrics for both classes
        if len(precision_per_class) == 2:
            precision_false, precision_true = precision_per_class
            recall_false, recall_true = recall_per_class
            f1_false, f1_true = f1_per_class
        else:
            # Handle case where only one class is present
            unique_classes = np.unique(y_true)
            if 0 in unique_classes and 1 not in unique_classes:
                precision_false, precision_true = precision_per_class[0], 0.0
                recall_false, recall_true = recall_per_class[0], 0.0
                f1_false, f1_true = f1_per_class[0], 0.0
            elif 1 in unique_classes and 0 not in unique_classes:
                precision_false, precision_true = 0.0, precision_per_class[0]
                recall_false, recall_true = 0.0, recall_per_class[0]
                f1_false, f1_true = 0.0, f1_per_class[0]
            else:
                precision_false = precision_true = 0.0
                recall_false = recall_true = 0.0
                f1_false = f1_true = 0.0
        
        # Aggregate metrics
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision_true': precision_true,
            'recall_true': recall_true,
            'f1_true': f1_true,
            'precision_false': precision_false,
            'recall_false': recall_false,
            'f1_false': f1_false,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'n_samples': len(y_true)
        }
    
    def evaluate_model(self, predictions_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            predictions_df (pd.DataFrame): Model predictions.
            model_name (str): Name of the model.
            
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        if len(predictions_df) == 0:
            logger.warning(f"No valid predictions for {model_name}")
            return {'metrics': self.compute_metrics(np.array([]), np.array([])), 'confusion_matrix': None}
        
        y_true = predictions_df['true_label'].values
        y_pred = predictions_df['prediction'].values
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Additional analysis
        success_rate = predictions_df['prediction'].notna().mean()
        
        result = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'success_rate': success_rate,
            'classification_report': classification_report(y_true, y_pred, target_names=['False', 'True'], output_dict=True, zero_division=0)
        }
        
        logger.info(f"{model_name} evaluation completed")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1-True: {metrics['f1_true']:.3f}")
        logger.info(f"  F1-False: {metrics['f1_false']:.3f}")
        logger.info(f"  Success Rate: {success_rate:.3f}")
        
        return result
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create comparison table of model performance.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Results for all models.
            
        Returns:
            pd.DataFrame: Comparison table.
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (True)': metrics['precision_true'],
                'Recall (True)': metrics['recall_true'],
                'F1-True': metrics['f1_true'],
                'Precision (False)': metrics['precision_false'],
                'Recall (False)': metrics['recall_false'],
                'F1-False': metrics['f1_false'],
                'Macro F1': metrics['macro_f1'],
                'N Samples': metrics['n_samples']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-True (descending)
        comparison_df = comparison_df.sort_values('F1-True', ascending=False)
        
        return comparison_df
    
    def generate_report(self, model_results: Dict[str, Dict[str, Any]], output_dir: str = "reports") -> None:
        """
        Generate evaluation report.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Results for all models.
            output_dir (str): Output directory for reports.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate comparison table
        comparison_df = self.compare_models(model_results)
        
        # Save comparison CSV
        comparison_df.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(comparison_df, model_results)
        
        # Save markdown report
        with open(output_path / "metrics.md", "w") as f:
            f.write(markdown_report)
        
        # Save detailed results as JSON
        with open(output_path / "detailed_results.json", "w") as f:
            json.dump(model_results, f, indent=2)
        
        logger.info(f"Reports saved to {output_dir}/")
    
    def _generate_markdown_report(self, comparison_df: pd.DataFrame, model_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate markdown evaluation report.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison table.
            model_results (Dict[str, Dict[str, Any]]): Detailed results.
            
        Returns:
            str: Markdown report content.
        """
        report = """# Financial Misinformation Detection - Model Evaluation Report

## Overview

This report presents the evaluation results for three single-LLM approaches to financial misinformation detection:

1. **Zero-shot Standard**: Simple task instruction + claim only
2. **Zero-shot Chain-of-Thought**: "Let's think step-by-step" reasoning
3. **Few-shot Chain-of-Thought**: 8 in-context examples with reasoning

## Performance Summary

"""
        
        # Add comparison table
        report += "### Model Comparison\n\n"
        report += comparison_df.round(3).to_markdown(index=False)
        report += "\n\n"
        
        # Add detailed results for each model
        report += "## Detailed Results\n\n"
        
        for model_name, results in model_results.items():
            if results['metrics']['n_samples'] == 0:
                continue
                
            metrics = results['metrics']
            cm = results['confusion_matrix']
            
            report += f"### {model_name}\n\n"
            report += f"- **Sample Size**: {metrics['n_samples']}\n"
            report += f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
            report += f"- **F1-True (Real Claims)**: {metrics['f1_true']:.3f}\n"
            report += f"- **F1-False (Fake Claims)**: {metrics['f1_false']:.3f}\n"
            report += f"- **Macro F1**: {metrics['macro_f1']:.3f}\n\n"
            
            if cm:
                report += "**Confusion Matrix:**\n\n"
                report += f"|       | Pred False | Pred True |\n"
                report += f"|-------|------------|-----------||\n"
                report += f"| **True False** | {cm[0][0]} | {cm[0][1]} |\n"
                report += f"| **True True**  | {cm[1][0]} | {cm[1][1]} |\n\n"
        
        # Add methodology section
        report += """## Methodology

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
"""
        
        return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Financial Misinformation Detection Models")
    parser.add_argument("--standard", default="preds_standard.csv", help="Zero-shot standard predictions")
    parser.add_argument("--cot", default="preds_cot.csv", help="Zero-shot CoT predictions")
    parser.add_argument("--fewshot", default="preds_fewshot.csv", help="Few-shot CoT predictions")
    parser.add_argument("--output", default="reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Model configurations
    models = {
        "Zero-shot Standard": args.standard,
        "Zero-shot Chain-of-Thought": args.cot,
        "Few-shot Chain-of-Thought": args.fewshot
    }
    
    # Evaluate each model
    model_results = {}
    for model_name, pred_file in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        predictions_df = evaluator.load_predictions(pred_file, model_name)
        results = evaluator.evaluate_model(predictions_df, model_name)
        model_results[model_name] = results
    
    # Generate comparison and reports
    logger.info("Generating evaluation reports...")
    evaluator.generate_report(model_results, args.output)
    
    # Print summary
    comparison_df = evaluator.compare_models(model_results)
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.round(3).to_string(index=False))
    print("="*60)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main() 