#!/usr/bin/env python3
"""
Data pipeline for Financial Misinformation Detection.
Loads finfact.json and creates stratified train/val/test splits.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the finfact.json dataset.
    
    Args:
        file_path (str): Path to the finfact.json file.
        
    Returns:
        pd.DataFrame: Loaded dataset as DataFrame.
    """
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    logger.info(f"Dataset loaded: {len(df)} records")
    
    # Log label distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
    
    return df


def preprocess_for_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataset for binary classification (True/False).
    Filters out NEI (Not Enough Information) labels.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: Preprocessed dataset with binary labels.
    """
    logger.info("Preprocessing for binary classification")
    
    # Filter out NEI labels for binary classification
    binary_df = df[df['label'].isin(['true', 'false'])].copy()
    
    # Convert to binary labels (1 for true, 0 for false)
    binary_df['binary_label'] = (binary_df['label'] == 'true').astype(int)
    
    logger.info(f"Binary dataset: {len(binary_df)} records after filtering out NEI")
    logger.info(f"Binary label distribution: {binary_df['binary_label'].value_counts().to_dict()}")
    
    return binary_df


def create_splits(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, 
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.
    
    Args:
        df (pd.DataFrame): Dataset to split.
        test_size (float): Test set proportion.
        val_size (float): Validation set proportion.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test splits.
    """
    logger.info(f"Creating splits: train={1-test_size-val_size:.2f}, val={val_size:.2f}, test={test_size:.2f}")
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['binary_label']
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val['binary_label']
    )
    
    logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Verify stratification
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        dist = split_df['binary_label'].value_counts(normalize=True)
        logger.info(f"{split_name} label distribution: {dist.to_dict()}")
    
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, 
               output_dir: str = "data") -> None:
    """
    Save the train/val/test splits to CSV files.
    
    Args:
        train (pd.DataFrame): Training split.
        val (pd.DataFrame): Validation split.
        test (pd.DataFrame): Test split.
        output_dir (str): Directory to save files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save splits
    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    
    logger.info(f"Splits saved to {output_dir}/")


def get_few_shot_examples(train_df: pd.DataFrame, n_examples: int = 8) -> Dict[str, Any]:
    """
    Select balanced few-shot examples from training data.
    
    Args:
        train_df (pd.DataFrame): Training data.
        n_examples (int): Total number of examples (should be even for balance).
        
    Returns:
        Dict[str, Any]: Selected examples with metadata.
    """
    logger.info(f"Selecting {n_examples} few-shot examples")
    
    # Ensure even number for balance
    n_per_class = n_examples // 2
    
    # Filter by claim length (15-25 tokens as per planning)
    train_df['claim_length'] = train_df['claim'].str.split().str.len()
    medium_length = train_df[(train_df['claim_length'] >= 15) & (train_df['claim_length'] <= 25)]
    
    # Get diverse issues if available
    examples = []
    
    for label in [0, 1]:  # false, true
        label_data = medium_length[medium_length['binary_label'] == label]
        
        if len(label_data) < n_per_class:
            # Fallback to all training data if not enough medium-length examples
            label_data = train_df[train_df['binary_label'] == label]
        
        # Sample examples
        selected = label_data.sample(n=min(n_per_class, len(label_data)), random_state=42)
        examples.extend(selected.to_dict('records'))
    
    logger.info(f"Selected {len(examples)} examples for few-shot prompting")
    
    return {
        'examples': examples,
        'n_true': sum(1 for ex in examples if ex['binary_label'] == 1),
        'n_false': sum(1 for ex in examples if ex['binary_label'] == 0),
        'avg_length': np.mean([len(ex['claim'].split()) for ex in examples])
    }


def main():
    """Main data pipeline execution."""
    logger.info("Starting data pipeline")
    
    # Load dataset
    df = load_dataset("finfact.json")
    
    # Preprocess for binary classification
    binary_df = preprocess_for_binary_classification(df)
    
    # Create splits
    train, val, test = create_splits(binary_df)
    
    # Save splits
    save_splits(train, val, test)
    
    # Generate few-shot examples
    few_shot_examples = get_few_shot_examples(train)
    
    # Save few-shot examples
    with open("data/few_shot_examples.json", "w") as f:
        json.dump(few_shot_examples, f, indent=2)
    
    logger.info("Data pipeline completed successfully")
    

if __name__ == "__main__":
    main() 