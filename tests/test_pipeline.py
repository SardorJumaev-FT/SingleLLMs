#!/usr/bin/env python3
"""
Unit tests for Financial Misinformation Detection pipeline.
Tests all major components: data pipeline, models, evaluation.
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_pipeline import (
    load_dataset, preprocess_for_binary_classification, 
    create_splits, get_few_shot_examples
)
from scripts.evaluate import ModelEvaluator


class TestDataPipeline(unittest.TestCase):
    """Test cases for data pipeline functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = [
            {
                "claim": "The stock market crashed yesterday",
                "label": "false",
                "author": "Test Author 1",
                "posted": "01/01/2020",
                "issues": ["market", "stocks"],
                "url": "http://test1.com",
                "sci_digest": ["Test digest"],
                "justification": "Test justification",
                "image_data": [],
                "evidence": []
            },
            {
                "claim": "Interest rates will increase next month according to the Fed",
                "label": "true", 
                "author": "Test Author 2",
                "posted": "02/01/2020",
                "issues": ["federal", "interest"],
                "url": "http://test2.com",
                "sci_digest": ["Test digest 2"],
                "justification": "Test justification 2",
                "image_data": [],
                "evidence": []
            },
            {
                "claim": "This financial claim has insufficient evidence",
                "label": "NEI",
                "author": "Test Author 3", 
                "posted": "03/01/2020",
                "issues": ["general"],
                "url": "http://test3.com",
                "sci_digest": ["Test digest 3"],
                "justification": "Test justification 3",
                "image_data": [],
                "evidence": []
            }
        ]
        
        # Create temporary test file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_file.name)
    
    def test_load_dataset(self):
        """Test dataset loading functionality."""
        df = load_dataset(self.temp_file.name)
        
        self.assertEqual(len(df), 3)
        self.assertIn('claim', df.columns)
        self.assertIn('label', df.columns)
        self.assertEqual(set(df['label'].unique()), {'false', 'true', 'NEI'})
    
    def test_preprocess_for_binary_classification(self):
        """Test binary classification preprocessing."""
        df = load_dataset(self.temp_file.name)
        binary_df = preprocess_for_binary_classification(df)
        
        self.assertEqual(len(binary_df), 2)  # Should exclude NEI
        self.assertIn('binary_label', binary_df.columns)
        self.assertEqual(set(binary_df['binary_label'].unique()), {0, 1})
        
        # Check label mapping
        true_row = binary_df[binary_df['label'] == 'true'].iloc[0]
        false_row = binary_df[binary_df['label'] == 'false'].iloc[0]
        self.assertEqual(true_row['binary_label'], 1)
        self.assertEqual(false_row['binary_label'], 0)
    
    def test_create_splits(self):
        """Test train/validation/test split creation."""
        # Create larger test dataset for splitting
        larger_data = self.test_data * 20  # 60 records
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(larger_data, f)
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path)
            binary_df = preprocess_for_binary_classification(df)
            
            train, val, test = create_splits(binary_df, test_size=0.2, val_size=0.2, random_state=42)
            
            # Check split sizes (approximately)
            total_size = len(binary_df)
            self.assertAlmostEqual(len(test) / total_size, 0.2, delta=0.1)
            self.assertAlmostEqual(len(val) / total_size, 0.2, delta=0.1)
            self.assertAlmostEqual(len(train) / total_size, 0.6, delta=0.1)
            
            # Check no overlap
            train_indices = set(train.index)
            val_indices = set(val.index)
            test_indices = set(test.index)
            
            self.assertEqual(len(train_indices & val_indices), 0)
            self.assertEqual(len(train_indices & test_indices), 0)
            self.assertEqual(len(val_indices & test_indices), 0)
            
        finally:
            os.unlink(temp_path)
    
    def test_get_few_shot_examples(self):
        """Test few-shot example selection."""
        # Create test data with varying claim lengths
        varied_data = []
        for i in range(20):
            if i < 10:
                claim = " ".join(["word"] * (15 + i))  # 15-24 words
                label = "true" if i % 2 == 0 else "false"
            else:
                claim = " ".join(["word"] * (10 + i))  # Longer claims
                label = "true" if i % 2 == 0 else "false"
            
            varied_data.append({
                "claim": claim,
                "label": label,
                "author": f"Author {i}",
                "posted": "01/01/2020",
                "issues": ["test"],
                "url": f"http://test{i}.com",
                "sci_digest": ["Test"],
                "justification": "Test",
                "image_data": [],
                "evidence": []
            })
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(varied_data, f)
            temp_path = f.name
        
        try:
            df = load_dataset(temp_path)
            binary_df = preprocess_for_binary_classification(df)
            
            examples = get_few_shot_examples(binary_df, n_examples=8)
            
            self.assertEqual(len(examples['examples']), 8)
            self.assertEqual(examples['n_true'], 4)
            self.assertEqual(examples['n_false'], 4)
            self.assertIsInstance(examples['avg_length'], float)
            
        finally:
            os.unlink(temp_path)


class TestModelEvaluator(unittest.TestCase):
    """Test cases for model evaluation functionality."""
    
    def setUp(self):
        """Set up test evaluation data."""
        self.evaluator = ModelEvaluator()
        
        # Create test predictions data
        np.random.seed(42)
        n_samples = 100
        
        self.test_predictions = pd.DataFrame({
            'prediction': np.random.choice([0, 1], n_samples),
            'true_label': np.random.choice([0, 1], n_samples),
            'claim': [f"Test claim {i}" for i in range(n_samples)],
            'index': range(n_samples)
        })
        
        # Create temporary prediction file
        self.temp_pred_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.test_predictions.to_csv(self.temp_pred_file.name, index=False)
        self.temp_pred_file.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_pred_file.name)
    
    def test_load_predictions(self):
        """Test prediction loading."""
        df = self.evaluator.load_predictions(self.temp_pred_file.name, "Test Model")
        
        self.assertEqual(len(df), 100)
        self.assertIn('prediction', df.columns)
        self.assertIn('true_label', df.columns)
        
        # Test with non-existent file
        empty_df = self.evaluator.load_predictions("non_existent.csv", "Test Model")
        self.assertTrue(empty_df.empty)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        metrics = self.evaluator.compute_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['f1_true'], 1.0)
        self.assertEqual(metrics['f1_false'], 1.0)
        self.assertEqual(metrics['n_samples'], 4)
        
        # Test with empty arrays
        empty_metrics = self.evaluator.compute_metrics(np.array([]), np.array([]))
        self.assertEqual(empty_metrics['accuracy'], 0.0)
        self.assertEqual(empty_metrics['n_samples'], 0)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        results = self.evaluator.evaluate_model(self.test_predictions, "Test Model")
        
        self.assertIn('metrics', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('success_rate', results)
        self.assertIn('classification_report', results)
        
        # Check metrics structure
        metrics = results['metrics']
        expected_keys = ['accuracy', 'precision_true', 'recall_true', 'f1_true',
                        'precision_false', 'recall_false', 'f1_false', 'macro_f1',
                        'weighted_f1', 'n_samples']
        
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock results for multiple models
        model_results = {
            "Model A": {
                'metrics': {
                    'accuracy': 0.8,
                    'precision_true': 0.75,
                    'recall_true': 0.85,
                    'f1_true': 0.8,
                    'precision_false': 0.85,
                    'recall_false': 0.75,
                    'f1_false': 0.8,
                    'macro_f1': 0.8,
                    'n_samples': 100
                }
            },
            "Model B": {
                'metrics': {
                    'accuracy': 0.75,
                    'precision_true': 0.7,
                    'recall_true': 0.8,
                    'f1_true': 0.75,
                    'precision_false': 0.8,
                    'recall_false': 0.7,
                    'f1_false': 0.75,
                    'macro_f1': 0.75,
                    'n_samples': 100
                }
            }
        }
        
        comparison_df = self.evaluator.compare_models(model_results)
        
        self.assertEqual(len(comparison_df), 2)
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('Accuracy', comparison_df.columns)
        self.assertIn('F1-True', comparison_df.columns)
        
        # Check if sorted by F1-True (descending)
        f1_values = comparison_df['F1-True'].values
        self.assertTrue(all(f1_values[i] >= f1_values[i+1] for i in range(len(f1_values)-1)))


class TestResponseParsing(unittest.TestCase):
    """Test cases for response parsing functionality."""
    
    def setUp(self):
        """Set up test cases for response parsing."""
        # Import parsing functions (these would be in the actual model scripts)
        # For now, we'll define simple versions for testing
        pass
    
    def test_standard_response_parsing(self):
        """Test parsing of standard model responses."""
        test_cases = [
            ("True", 1),
            ("False", 0),
            ("true", 1),
            ("false", 0),
            ("TRUE", 1),
            ("FALSE", 0),
            ("The answer is True", 1),
            ("I believe this is False", 0),
            ("Unclear response", None)
        ]
        
        def parse_response(response):
            """Simple response parser for testing."""
            response_lower = response.lower().strip()
            if "true" in response_lower and "false" not in response_lower:
                return 1
            elif "false" in response_lower and "true" not in response_lower:
                return 0
            return None
        
        for response, expected in test_cases:
            with self.subTest(response=response):
                result = parse_response(response)
                self.assertEqual(result, expected)
    
    def test_cot_response_parsing(self):
        """Test parsing of Chain-of-Thought responses."""
        cot_response = """
        Analysis: Let me think step-by-step.
        
        First, I'll analyze the key financial concepts: This involves market analysis.
        
        Next, I'll check the factual accuracy: The numbers seem accurate.
        
        Then, I'll evaluate against financial knowledge: This aligns with known principles.
        
        Finally, checking for logical consistency: The claim is consistent.
        
        Based on this analysis, my determination is: True
        """
        
        def parse_cot_response(response):
            """Simple CoT response parser for testing."""
            import re
            final_pattern = r"based on this analysis.*?my determination is:?\s*(\[?(?:true|false)\]?)"
            match = re.search(final_pattern, response.lower())
            
            if match:
                final_answer = match.group(1).strip("[]").lower()
                if "true" in final_answer:
                    return 1
                elif "false" in final_answer:
                    return 0
            return None
        
        result = parse_cot_response(cot_response)
        self.assertEqual(result, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from data loading to evaluation."""
        # Create minimal test dataset
        test_data = [
            {
                "claim": f"Financial claim number {i}",
                "label": "true" if i % 2 == 0 else "false",
                "author": f"Author {i}",
                "posted": "01/01/2020",
                "issues": ["test"],
                "url": f"http://test{i}.com",
                "sci_digest": ["Test"],
                "justification": "Test",
                "image_data": [],
                "evidence": []
            }
            for i in range(20)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Test data pipeline
            df = load_dataset(temp_path)
            binary_df = preprocess_for_binary_classification(df)
            train, val, test = create_splits(binary_df, random_state=42)
            few_shot_examples = get_few_shot_examples(train)
            
            # Verify pipeline components work together
            self.assertGreater(len(train), 0)
            self.assertGreater(len(val), 0)
            self.assertGreater(len(test), 0)
            self.assertEqual(len(few_shot_examples['examples']), 8)
            
            # Test evaluation setup
            evaluator = ModelEvaluator()
            
            # Create mock predictions matching test set
            mock_predictions = pd.DataFrame({
                'prediction': np.random.choice([0, 1], len(test)),
                'true_label': test['binary_label'].values,
                'claim': test['claim'].values,
                'index': test.index
            })
            
            results = evaluator.evaluate_model(mock_predictions, "Test Model")
            self.assertIn('metrics', results)
            self.assertGreater(results['metrics']['n_samples'], 0)
            
        finally:
            os.unlink(temp_path)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataPipeline,
        TestModelEvaluator,
        TestResponseParsing,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 