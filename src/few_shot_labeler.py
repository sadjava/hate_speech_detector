#!/usr/bin/env python3

import pandas as pd
import requests
import json
import random
from typing import List, Tuple
import time
import argparse
from pathlib import Path
import os

class FewShotLabeler:
    def __init__(self, api_key: str = None, delay: float = 2.0):
        # OpenRouter API configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Get free key at https://openrouter.ai and set OPENROUTER_API_KEY environment variable")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_name = "mistralai/mistral-7b-instruct:free"  # Free model
        self.delay = delay  # Configurable delay between requests
        
        self.categories = [
            "Race", "Sexual Orientation", "Gender", "Physical Appearance", 
            "Religion", "Class", "Disability", "Appropriate"
        ]
        
        print(f"Using OpenRouter with free model: {self.model_name}")
        print(f"Rate limiting: {self.delay} seconds between requests")
    
    def load_labeled_data(self, labeled_file_path: str, n_examples_per_category: int = 3) -> List[Tuple[str, str]]:
        """Load manually labeled examples for few-shot prompting"""
        try:
            labeled_data = pd.read_csv(labeled_file_path)
            print(f"Loaded {len(labeled_data)} labeled examples from {labeled_file_path}")
            
            # Verify required columns
            if 'text' not in labeled_data.columns or 'category' not in labeled_data.columns:
                raise ValueError("Labeled data must have 'text' and 'category' columns")
                
        except Exception as e:
            raise ValueError(f"Error loading labeled data: {e}")
        
        # Create few-shot examples
        examples = []
        
        print(f"Available categories in labeled data: {labeled_data['category'].unique()}")
        
        for category in self.categories:
            category_data = labeled_data[labeled_data['category'] == category]
            if len(category_data) > 0:
                # Sample examples from this category
                n_sample = min(n_examples_per_category, len(category_data))
                sampled = category_data.sample(n=n_sample, random_state=42)
                for _, row in sampled.iterrows():
                    examples.append((row['text'], row['category']))
                print(f"  {category}: {n_sample} examples")
            else:
                print(f"  {category}: 0 examples (not found in data)")
        
        print(f"Total few-shot examples: {len(examples)}")
        return examples
    
    def create_classification_prompt(self, examples: List[Tuple[str, str]], text_to_classify: str, language: str = "en") -> List[dict]:
        """Create messages for OpenRouter API with few-shot examples"""
        
        if language == "en":
            system_content = f"""You are an expert at classifying hate speech. Classify the given text into exactly one of these categories:
{', '.join(self.categories)}

Guidelines:
- Race: Content targeting racial groups with slurs, stereotypes, or discrimination
- Sexual Orientation: Content targeting LGBTQ+ individuals with slurs or discrimination  
- Gender: Content with sexist attacks or gender-based discrimination
- Physical Appearance: Content targeting physical traits, body shaming, appearance-based attacks
- Religion: Content targeting religious beliefs, groups, or practices
- Class: Content targeting socioeconomic status or class-based discrimination
- Disability: Content targeting people with disabilities or using disability slurs
- Appropriate: Content that doesn't contain hate speech or discrimination

Examples:"""
            
        else:  # Russian
            system_content = f"""Вы эксперт по классификации речи ненависти. Классифицируйте данный текст в одну из категорий:
{', '.join(self.categories)}

Примеры:"""
        
        # Add few-shot examples to system message
        for text, category in examples[:15]:  # Limit examples to avoid token limits
            system_content += f"\nText: \"{text}\" → Category: {category}"
        
        system_content += "\n\nRespond with ONLY the category name, nothing else."
        
        # Create user message
        if language == "en":
            user_content = f"Classify this text:\nText: \"{text_to_classify}\"\nCategory:"
        else:
            user_content = f"Классифицируйте этот текст:\nТекст: \"{text_to_classify}\"\nКатегория:"
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def classify_text_with_retry(self, text: str, examples: List[Tuple[str, str]], language: str = "en", max_retries: int = 5) -> str:
        """Classify text with exponential backoff retry logic"""
        
        for attempt in range(max_retries):
            try:
                messages = self.create_classification_prompt(examples, text, language)
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",  # Required for OpenRouter
                    "X-Title": "Hate Speech Classifier"
                }
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 20,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    print(f"Text: {text}")
                    print(f"Prediction: {prediction}")
                    print("-" * 100)
                    
                    # Clean up the prediction
                    prediction = prediction.replace("Category:", "").strip()
                    
                    # Validate prediction is in our categories
                    if prediction in self.categories:
                        return prediction
                    else:
                        # Try partial matching
                        for category in self.categories:
                            if category.lower() in prediction.lower():
                                return category
                        
                        print(f"Invalid prediction '{prediction}', defaulting to 'Appropriate'")
                        return "Appropriate"
                        
                elif response.status_code == 401:
                    print("Authentication failed. Check your OpenRouter API key.")
                    print("Get a free key at: https://openrouter.ai")
                    return "Appropriate"
                    
                elif response.status_code == 429:
                    # Rate limit exceeded - exponential backoff
                    wait_time = (2 ** attempt) * self.delay  # Exponential backoff
                    print(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue  # Retry
                    
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        wait_time = self.delay * (attempt + 1)
                        print(f"Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        continue
                    return "Appropriate"
                    
            except Exception as e:
                print(f"Error classifying text (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = self.delay * (attempt + 1)
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                return "Appropriate"
        
        # All retries failed
        print(f"All {max_retries} attempts failed for text: {text[:50]}...")
        return "Appropriate"
    
    def process_file(self, labeled_file_path: str, unlabeled_file_path: str, 
                    output_file_path: str, language: str = "en", 
                    n_examples_per_category: int = 3, batch_size: int = 5, save_frequency: int = 20):
        """Process unlabeled data file and add category predictions with batching and auto-save"""
        
        # Load examples from labeled data
        print(f"Loading few-shot examples from {labeled_file_path}...")
        examples = self.load_labeled_data(labeled_file_path, n_examples_per_category)
        
        if not examples:
            raise ValueError("No valid examples found in labeled data")
        
        # Load unlabeled data
        print(f"Loading unlabeled data from {unlabeled_file_path}...")
        try:
            unlabeled_data = pd.read_csv(unlabeled_file_path)
            print(f"Loaded {len(unlabeled_data)} texts to classify")
            
            if 'text' not in unlabeled_data.columns:
                raise ValueError("Unlabeled data must have 'text' column")
                
        except Exception as e:
            raise ValueError(f"Error loading unlabeled data: {e}")
        
        # Check if we're resuming from a previous run
        if os.path.exists(output_file_path):
            try:
                existing_results = pd.read_csv(output_file_path)
                if 'category' in existing_results.columns and len(existing_results) > 0:
                    start_index = len(existing_results)
                    print(f"Resuming from index {start_index} (found existing results)")
                    predicted_categories = existing_results['category'].tolist()
                else:
                    start_index = 0
                    predicted_categories = []
            except:
                start_index = 0
                predicted_categories = []
        else:
            start_index = 0
            predicted_categories = []
        
        total_texts = len(unlabeled_data)
        
        print(f"Starting classification from index {start_index}/{total_texts}...")
        print(f"Using model: {self.model_name}")
        print(f"Batch size: {batch_size} texts, then {self.delay * batch_size:.1f}s delay")
        
        batch_count = 0
        
        for i in range(start_index, total_texts):
            row = unlabeled_data.iloc[i]
            text = str(row['text']).strip()
            
            # Skip empty texts
            if not text or text.lower() == 'nan':
                predicted_categories.append("Appropriate")
                continue
            
            # Show progress
            if i % 10 == 0:
                print(f"Progress: {i+1}/{total_texts} ({((i+1)/total_texts)*100:.1f}%)")
            
            # Classify the text
            category = self.classify_text_with_retry(text, examples, language)
            predicted_categories.append(category)
            
            batch_count += 1
            
            # Batch processing with longer delays
            if batch_count >= batch_size:
                batch_delay = self.delay * batch_size
                print(f"Completed batch. Waiting {batch_delay:.1f} seconds to respect rate limits...")
                time.sleep(batch_delay)
                batch_count = 0
            else:
                # Short delay between individual requests
                time.sleep(self.delay)
            
            # Auto-save progress periodically
            if (i + 1) % save_frequency == 0:
                self.save_progress(unlabeled_data, predicted_categories, output_file_path, i + 1)
        
        # Final save
        self.save_results(unlabeled_data, predicted_categories, output_file_path, total_texts)
    
    def save_progress(self, unlabeled_data: pd.DataFrame, predicted_categories: List[str], 
                     output_file_path: str, processed_count: int):
        """Save progress to avoid losing work"""
        result_data = unlabeled_data.iloc[:processed_count].copy()
        result_data['category'] = predicted_categories
        result_data.to_csv(output_file_path, index=False)
        print(f"Progress saved: {processed_count} texts classified")
    
    def save_results(self, unlabeled_data: pd.DataFrame, predicted_categories: List[str], 
                    output_file_path: str, total_texts: int):
        """Save final results with summary"""
        result_data = unlabeled_data.copy()
        result_data['category'] = predicted_categories
        
        # Save results
        print(f"Saving final results to {output_file_path}...")
        result_data.to_csv(output_file_path, index=False)
        
        # Print summary
        category_counts = pd.Series(predicted_categories).value_counts()
        print(f"\nClassification Summary:")
        for category, count in category_counts.items():
            percentage = (count / len(predicted_categories)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print(f"\nCompleted! Results saved to: {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Label hate speech data using few-shot learning with OpenRouter free models")
    parser.add_argument("--labeled", required=True, help="CSV file with manually labeled examples (must have 'text' and 'category' columns)")
    parser.add_argument("--unlabeled", required=True, help="CSV file with unlabeled data to classify (must have 'text' column)")
    parser.add_argument("--output", required=True, help="Output CSV file with added 'category' column")
    parser.add_argument("--language", choices=["en", "ru"], default="en", help="Language (en/ru)")
    parser.add_argument("--examples", type=int, default=3, help="Examples per category for few-shot learning")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY environment variable)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between requests in seconds (default: 3.0)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size before longer delay (default: 5)")
    parser.add_argument("--save-frequency", type=int, default=20, help="Save progress every N classifications (default: 20)")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.labeled).exists():
        print(f"Error: Labeled file not found: {args.labeled}")
        return 1
    
    if not Path(args.unlabeled).exists():
        print(f"Error: Unlabeled file not found: {args.unlabeled}")
        return 1
    
    try:
        # Initialize labeler with custom delay
        labeler = FewShotLabeler(api_key=args.api_key, delay=args.delay)
        
        # Process the files
        labeler.process_file(
            labeled_file_path=args.labeled,
            unlabeled_file_path=args.unlabeled,
            output_file_path=args.output,
            language=args.language,
            n_examples_per_category=args.examples,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTips for rate limiting:")
        print("- Increase --delay (try 5.0 or 10.0 seconds)")
        print("- Reduce --batch-size (try 3 or 1)")
        print("- Use different free models")
        print("- Consider upgrading to paid tier")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())