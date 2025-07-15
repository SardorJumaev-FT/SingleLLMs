#!/usr/bin/env python3
"""
Zero-shot Chain-of-Thought LLM for Financial Misinformation Detection.
Uses "Let's think step-by-step" reasoning approach.
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


class ZeroShotCoTClassifier:
    """Zero-shot Chain-of-Thought classifier for financial misinformation."""
    
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
        with open("prompts/cot_zero.txt", "r") as f:
            self.prompt_template = f.read().strip()
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model}")
    
    def classify_claim(self, claim: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Classify a single financial claim using Chain-of-Thought.
        
        Args:
            claim (str): The claim to classify.
            max_retries (int): Maximum number of API retries.
            
        Returns:
            Dict[str, Any]: Classification result with prediction and reasoning.
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
                    max_tokens=500,  # Allow longer response for reasoning
                    timeout=60
                )
                
                raw_response = response.choices[0].message.content.strip()
                
                # Parse response to extract True/False and reasoning
                prediction, reasoning = self._parse_cot_response(raw_response)
                
                return {
                    "prediction": prediction,
                    "reasoning": reasoning,
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
                        "reasoning": None,
                        "raw_response": f"ERROR: {str(e)}",
                        "model": current_model,
                        "temperature": self.temperature,
                        "attempt": attempt + 1,
                        "tokens_used": None
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_cot_response(self, response: str) -> tuple:
        """
        Parse the Chain-of-Thought response to extract prediction and reasoning.
        
        Args:
            response (str): Raw model response.
            
        Returns:
            tuple: (prediction, reasoning)
        """
        # Extract the final determination
        prediction = None
        reasoning = response
        
        # Look for the final determination pattern
        final_pattern = r"based on this analysis.*?my determination is:?\s*(\[?(?:true|false)\]?)"
        match = re.search(final_pattern, response.lower())
        
        if match:
            final_answer = match.group(1).strip("[]").lower()
            if "true" in final_answer:
                prediction = 1
            elif "false" in final_answer:
                prediction = 0
        else:
            # Fallback: look for True/False anywhere in response
            response_lower = response.lower()
            if "true" in response_lower and "false" not in response_lower:
                prediction = 1
            elif "false" in response_lower and "true" not in response_lower:
                prediction = 0
        
        if prediction is None:
            logger.warning(f"Could not parse CoT response: {response[:100]}...")
        
        return prediction, reasoning
    
    def classify_dataset(self, df: pd.DataFrame, delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Classify all claims in a dataset using Chain-of-Thought.
        
        Args:
            df (pd.DataFrame): Dataset with 'claim' column.
            delay (float): Delay between API calls (rate limiting).
            
        Returns:
            List[Dict[str, Any]]: Classification results.
        """
        results = []
        
        logger.info(f"Classifying {len(df)} claims with Chain-of-Thought...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="CoT Classification"):
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
        
        logger.info("Chain-of-Thought classification completed")
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
        
        # Save reasoning separately for analysis
        reasoning_path = output_path.replace('.csv', '_reasoning.txt')
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                if result.get('reasoning'):
                    f.write(f"=== Example {i+1} ===\n")
                    f.write(f"Claim: {result.get('claim', 'N/A')}\n")
                    f.write(f"True Label: {result.get('true_label', 'N/A')}\n")
                    f.write(f"Prediction: {result.get('prediction', 'N/A')}\n")
                    f.write(f"Reasoning:\n{result['reasoning']}\n\n")
        
        logger.info(f"Reasoning saved to {reasoning_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Zero-shot CoT Financial Misinformation Classifier")
    parser.add_argument("--model", default="o4-mini", help="OpenAI model to use (fallback: gpt-4.1)")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--input", default="data/test.csv", help="Input CSV file")
    parser.add_argument("--output", default="preds_cot.csv", help="Output CSV file")
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
    classifier = ZeroShotCoTClassifier(model=args.model, temperature=args.temp)
    
    # Run classification
    results = classifier.classify_dataset(df, delay=args.delay)
    
    # Save results
    classifier.save_results(results, args.output)
    
    logger.info("Zero-shot Chain-of-Thought classification completed successfully")


if __name__ == "__main__":
    main() 