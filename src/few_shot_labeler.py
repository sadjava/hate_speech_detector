#!/usr/bin/env python3

import pandas as pd
import random
from typing import List, Tuple
import time
import argparse
import os
from enum import Enum
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

class HateSpeechCategory(str, Enum):
    """Enum for hate speech categories"""
    RACE = "Race"
    SEXUAL_ORIENTATION = "Sexual Orientation"
    GENDER = "Gender"
    PHYSICAL_APPEARANCE = "Physical Appearance"
    RELIGION = "Religion"
    CLASS = "Class"
    DISABILITY = "Disability"
    APPROPRIATE = "Appropriate"

class HateSpeechClassification(BaseModel):
    """Structured output for hate speech classification"""
    category: HateSpeechCategory = Field(
        description="The hate speech category that best fits the text"
    )

class FewShotLabeler:
    def __init__(self, api_key: str = None, delay: float = 1.0, model_name: str = "gpt-3.5-turbo"):
        # OpenAI API configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=100,
            timeout=30
        )
        
        self.delay = delay
        self.categories = [category.value for category in HateSpeechCategory]
        
        # Setup structured output parser
        self.parser = PydanticOutputParser(pydantic_object=HateSpeechClassification)
        
        # Rate limiting configuration
        self.max_retries = 5
        self.batch_size = 10
        self.batch_delay = 5
        self.save_interval = 20
        
        print(f"🤖 Initialized LangChain labeler with {model_name}")
        print(f"📋 Categories: {len(self.categories)} categories")
    
    def load_training_examples(self, filepath: str, max_examples: int = 16) -> List[Tuple[str, str]]:
        """Load manually labeled examples for few-shot learning"""
        print(f"📚 Loading training examples from {filepath}")
        
        df = pd.read_csv(filepath)
        required_columns = ['text', 'category']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Training file must have columns: {required_columns}")
        
        # Filter valid categories
        valid_examples = df[df['category'].isin(self.categories)].copy()
        
        if len(valid_examples) == 0:
            raise ValueError(f"No valid examples found. Categories must be one of: {self.categories}")
        
        # Balance examples across categories if possible
        examples = []
        for category in self.categories:
            cat_examples = valid_examples[valid_examples['category'] == category]
            if len(cat_examples) > 0:
                # Take up to 2 examples per category for better token efficiency
                sample_size = min(2, len(cat_examples))
                examples.extend([(row['text'], row['category']) for _, row in cat_examples.sample(sample_size).iterrows()])
        
        print(f"✅ Loaded {len(examples)} training examples across {len(set(ex[1] for ex in examples))} categories")
        return examples
    
    def create_few_shot_prompt(self, examples: List[Tuple[str, str]]) -> ChatPromptTemplate:
        """Create few-shot prompt template with structured output"""
        
        # Few-shot example template
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Classify this text: {input}"),
            ("ai", "{{\"category\": \"{output}\"}}")
        ])
        
        # Convert examples to format expected by LangChain
        few_shot_examples = [
            {"input": text, "output": category}
            for text, category in examples
        ]
        
        # Create few-shot prompt
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shot_examples
        )
        
        # Get format instructions and escape curly braces for LangChain
        format_instructions = self.parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        
        # Main classification prompt
        system_message = f"""You are an expert hate speech classifier. Classify text into exactly one of these categories:

{', '.join(self.categories)}

Category Guidelines:
- Race: Racial slurs, stereotypes, discrimination based on race
- Sexual Orientation: Homophobic content, discrimination against LGBTQ+
- Gender: Sexist content, misogyny, gender-based harassment
- Physical Appearance: Body shaming, lookism, appearance-based harassment
- Religion: Religious discrimination, islamophobia, antisemitism
- Class: Classist content, economic discrimination
- Disability: Ableist content, discrimination against disabled people
- Appropriate: Non-hateful content, normal conversation

{format_instructions}

Respond with valid JSON only."""
        
        # Complete prompt template
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            few_shot_prompt,
            ("human", "Classify this text: {text}")
        ])
        return final_prompt
    
    def classify_with_retry(self, text: str, prompt_template: ChatPromptTemplate) -> str:
        """Classify single text with structured output and retry logic"""
        
        # Create the chain
        chain = prompt_template | self.llm | self.parser
        
        for attempt in range(self.max_retries):
            try:
                # Run the chain
                result = chain.invoke({"text": text})
                
                # Extract category from structured output
                category = result.category.value
                
                print(f"✅ Category: {category}")
                return category
                
            except OutputParserException as e:
                print(f"⚠️ Parser error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return "Appropriate"
                time.sleep(2 ** attempt)
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate" in error_msg or "429" in error_msg:
                    # Rate limit - exponential backoff
                    wait_time = (2 ** attempt) * 3
                    print(f"⏳ Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                    
                elif "401" in error_msg or "auth" in error_msg:
                    raise ValueError("❌ OpenAI API authentication failed. Check your API key.")
                    
                else:
                    print(f"⚠️ API error (attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        return "Appropriate"
                    time.sleep(2 ** attempt)
        
        print(f"❌ Max retries exceeded, defaulting to 'Appropriate'")
        return "Appropriate"
    
    def save_progress(self, df: pd.DataFrame, output_file: str):
        """Save current progress"""
        df.to_csv(output_file, index=False)
        print(f"💾 Progress saved to {output_file}")
    
    def label_dataset(self, training_file: str, input_file: str, output_file: str, text_column: str = "text"):
        """Label entire dataset with few-shot learning and structured output"""
        
        print(f"🚀 Starting LangChain few-shot labeling with structured output")
        print(f"📊 Training file: {training_file}")
        print(f"📄 Input file: {input_file}")
        print(f"💾 Output file: {output_file}")
        print(f"⏱️ Delay: {self.delay}s between requests")
        
        # Load training examples
        examples = self.load_training_examples(training_file)
        
        # Create prompt template
        prompt_template = self.create_few_shot_prompt(examples)
        print(f"🎯 Created few-shot prompt with {len(examples)} examples")
        
        # Load input data
        print(f"📂 Loading input data from {input_file}")
        df = pd.read_csv(input_file)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in input file")
        
        # Check if we're resuming
        start_idx = 0
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            if 'category' in existing_df.columns:
                start_idx = len(existing_df[existing_df['category'].notna()])
                df = existing_df.copy()
                print(f"📋 Resuming from index {start_idx}")
        
        # Initialize category column if not exists
        if 'category' not in df.columns:
            df['category'] = None
        
        total_texts = len(df)
        processed = start_idx
        
        print(f"📝 Processing {total_texts - start_idx} texts (starting from {start_idx})")
        
        try:
            for i in range(start_idx, total_texts):
                text = str(df.iloc[i][text_column])
                
                # Skip if already labeled
                if pd.isna(df.iloc[i].get('category')) or df.iloc[i]['category'] == '':
                    print(f"🔍 [{i+1}/{total_texts}] Classifying: {text[:80]}...")
                    
                    category = self.classify_with_retry(text, prompt_template)
                    df.iloc[i, df.columns.get_loc('category')] = category
                    
                    print(f"📝 [{i+1}/{total_texts}] → {category}")
                    processed += 1
                    
                    # Rate limiting
                    time.sleep(self.delay)
                    
                    # Batch delay
                    if (i + 1) % self.batch_size == 0:
                        print(f"⏸️ Batch complete, waiting {self.batch_delay}s...")
                        time.sleep(self.batch_delay)
                    
                    # Save progress periodically
                    if (i + 1) % self.save_interval == 0:
                        self.save_progress(df, output_file)
                else:
                    print(f"⏭️ [{i+1}/{total_texts}] Already labeled: {df.iloc[i]['category']}")
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Interrupted by user. Saving progress...")
            self.save_progress(df, output_file)
            return
        
        # Final save
        self.save_progress(df, output_file)
        
        # Summary
        final_stats = df['category'].value_counts()
        print(f"\n📊 Final Results:")
        print(f"✅ Total processed: {processed}")
        for category, count in final_stats.items():
            print(f"   {category}: {count}")
        
        print(f"🎉 Labeling complete! Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="LangChain few-shot hate speech labeling with structured output")
    parser.add_argument("--training", required=True, help="CSV file with manually labeled examples (text,category columns)")
    parser.add_argument("--input", required=True, help="CSV file with texts to label")
    parser.add_argument("--output", required=True, help="Output CSV file with labels")
    parser.add_argument("--text-column", default="text", help="Name of text column in input file")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    try:
        labeler = FewShotLabeler(
            api_key=args.api_key,
            delay=args.delay,
            model_name=args.model
        )
        
        labeler.label_dataset(
            training_file=args.training,
            input_file=args.input,
            output_file=args.output,
            text_column=args.text_column
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())