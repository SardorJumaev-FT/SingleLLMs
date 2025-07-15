#!/usr/bin/env python3
"""
Zero-shot Standard Prompt LLM for Financial Misinformation Detection.
Uses simple task instruction + claim only.
"""

import argparse
import json
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    logger.error("OpenAI package not installed. Please run: pip install openai")
    exit(1)


class ZeroShotStandardClassifier:
    """Zero-shot standard prompt classifier for financial misinformation."""
    
    def __init__(self, model: str = "o4-mini", temperature: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            model (str): OpenAI model to use.
            temperature (float): Sampling temperature.
        """
        self.model = model
        self.fallback_model = "gpt-4.1"  # Fallback if primary model unavailable
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load prompt template
        with open("prompts/standard.txt", "r") as f:
            self.prompt_template = f.read().strip()
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model}")
    
    def classify_claim(self, claim: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Classify a single financial claim.
        
        Args:
            claim (str): The claim to classify.
            max_retries (int): Maximum number of API retries.
            
        Returns:
            Dict[str, Any]: Classification result with prediction and metadata.
        """
        prompt = self.prompt_template.format(claim=claim)
        
        current_model = self.model
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=10,  # Short response expected
                    timeout=30
                )
                
                raw_response = response.choices[0].message.content.strip()
                
                # Parse response to extract True/False
                prediction = self._parse_response(raw_response)
                
                return {
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "model": current_model,
                    "temperature": self.temperature,
                    "attempt": attempt + 1,
                    "tokens_used": response.usage.total_tokens if response.usage else None
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with {current_model}: {str(e)}")
                
                # Try fallback model if primary model fails and we haven't tried it yet
                if current_model == self.model and "model" in str(e).lower():
                    logger.info(f"Trying fallback model: {self.fallback_model}")
                    current_model = self.fallback_model
                    continue
                
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for claim: {claim[:50]}...")
                    return {
                        "prediction": None,
                        "raw_response": f"ERROR: {str(e)}",
                        "model": current_model,
                        "temperature": self.temperature,
                        "attempt": attempt + 1,
                        "tokens_used": None
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_response(self, response: str) -> int:
        """
        Parse the model response to extract binary prediction.
        
        Args:
            response (str): Raw model response.
            
        Returns:
            int: 1 for True, 0 for False, None for unparseable.
        """
        response_lower = response.lower().strip()
        
        # Look for explicit True/False
        if "true" in response_lower and "false" not in response_lower:
            return 1
        elif "false" in response_lower and "true" not in response_lower:
            return 0
        
        # Fallback patterns
        if re.search(r'\btrue\b', response_lower):
            return 1
        elif re.search(r'\bfalse\b', response_lower):
            return 0
        
        logger.warning(f"Could not parse response: {response}")
        return None
    
    def classify_dataset(self, df: pd.DataFrame, delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Classify all claims in a dataset.
        
        Args:
            df (pd.DataFrame): Dataset with 'claim' column.
            delay (float): Delay between API calls (rate limiting).
            
        Returns:
            List[Dict[str, Any]]: Classification results.
        """
        results = []
        
        logger.info(f"Classifying {len(df)} claims...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            claim = row['claim']
            result = self.classify_claim(claim)
            
            # Add metadata
            result['index'] = idx
            result['claim'] = claim
            result['true_label'] = row.get('binary_label', None)
            
            results.append(result)
            
            # Rate limiting
            if delay > 0:
                time.sleep(delay)
        
        logger.info("Classification completed")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save classification results to file.
        
        Args:
            results (List[Dict[str, Any]]): Classification results.
            output_path (str): Output file path.
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Save detailed results
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Log summary statistics
        if 'prediction' in df.columns:
            successful_predictions = df['prediction'].notna().sum()
            logger.info(f"Successful predictions: {successful_predictions}/{len(df)}")
            
            if 'true_label' in df.columns and df['true_label'].notna().any():
                # Quick accuracy calculation for successful predictions
                valid_mask = df['prediction'].notna() & df['true_label'].notna()
                if valid_mask.sum() > 0:
                    accuracy = (df.loc[valid_mask, 'prediction'] == df.loc[valid_mask, 'true_label']).mean()
                    logger.info(f"Quick accuracy: {accuracy:.3f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Zero-shot Standard Financial Misinformation Classifier")
    parser.add_argument("--model", default="o4-mini", help="OpenAI model to use (fallback: gpt-4.1)")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--input", default="data/test.csv", help="Input CSV file")
    parser.add_argument("--output", default="preds_standard.csv", help="Output CSV file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Load test data
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        exit(1)
    
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} test examples")
    
    # Initialize classifier
    classifier = ZeroShotStandardClassifier(model=args.model, temperature=args.temp)
    
    # Run classification
    results = classifier.classify_dataset(df, delay=args.delay)
    
    # Save results
    classifier.save_results(results, args.output)
    
    logger.info("Zero-shot standard classification completed successfully")


if __name__ == "__main__":
    main() 