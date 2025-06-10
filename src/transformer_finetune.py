#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Text augmentation imports
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.char as nac
import nlpaug.flow as nafc
import nltk
from nltk.corpus import wordnet
import random
import re
from googletrans import Translator
import time

import argparse
import os
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from collections import Counter
import torch.nn as nn

class HateSpeechCategory(str, Enum):
    """Enum for hate speech categories (matching your existing code)"""
    RACE = "Race"
    SEXUAL_ORIENTATION = "Sexual Orientation"
    GENDER = "Gender"
    PHYSICAL_APPEARANCE = "Physical Appearance"
    RELIGION = "Religion"
    CLASS = "Class"
    DISABILITY = "Disability"
    APPROPRIATE = "Appropriate"

class TextAugmenter:
    """Comprehensive text augmentation for hate speech data"""
    
    def __init__(self):
        self.setup_nltk()
        self.setup_augmenters()
        self.translator = None  # Initialize lazily
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print("📚 Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def setup_augmenters(self):
        """Initialize various augmenters"""
        try:
            # Word-level augmenters
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
            self.contextual_aug = naw.ContextualWordEmbsAug(
                model_path='distilbert-base-uncased', 
                action="substitute",
                aug_p=0.3
            )
            self.random_word_aug = naw.RandomWordAug(action="swap", aug_p=0.3)
            self.antonym_aug = naw.AntonymAug(aug_p=0.2)
            
            # Character-level augmenters
            self.keyboard_aug = nac.KeyboardAug(aug_char_p=0.1)
            self.ocr_aug = nac.OcrAug(aug_char_p=0.1)
            self.random_char_aug = nac.RandomCharAug(action="substitute", aug_char_p=0.1)
            
            # Sentence-level augmenters
            self.abs_summ_aug = nas.AbstSummAug(model_path='t5-base')
            
            print("✅ Text augmenters initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Some augmenters failed to initialize: {e}")
            # Fallback to simple augmenters
            self.setup_simple_augmenters()
    
    def setup_simple_augmenters(self):
        """Setup simple augmenters as fallback"""
        self.synonym_aug = None
        self.contextual_aug = None
        self.random_word_aug = None
        self.antonym_aug = None
        self.keyboard_aug = None
        self.ocr_aug = None
        self.random_char_aug = None
        self.abs_summ_aug = None
        print("🔧 Using simple augmentation fallbacks")
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms using WordNet"""
        words = text.split()
        if len(words) < 2:
            return text
            
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            if num_replaced >= n:
                break
                
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != random_word.lower():
                        synonyms.append(synonym)
            
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert n words into the sentence"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            # Get synonyms of random words
            random_word = random.choice([w for w in words if w.isalpha()])
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().replace('_', ' '))
            
            if synonyms:
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_synonym)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap two words n times"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            random_idx_1 = random.randint(0, len(words) - 1)
            random_idx_2 = random_idx_1
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(words) - 1)
                counter += 1
                if counter > 3:
                    break
            words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            # Return a random word if all words are deleted
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def back_translation(self, text: str, intermediate_lang: str = 'es') -> str:
        """Perform back translation via intermediate language"""
        try:
            if self.translator is None:
                self.translator = Translator()
            
            # Translate to intermediate language
            translated = self.translator.translate(text, dest=intermediate_lang)
            time.sleep(0.1)  # Rate limiting
            
            # Translate back to English
            back_translated = self.translator.translate(translated.text, dest='en')
            time.sleep(0.1)  # Rate limiting
            
            return back_translated.text
            
        except Exception as e:
            print(f"⚠️ Back translation failed: {e}")
            return text
    
    def easy_data_augmentation(self, text: str, alpha: float = 0.1, num_aug: int = 1) -> List[str]:
        """Easy Data Augmentation (EDA) technique"""
        augmented_texts = []
        words = text.split()
        num_words = len(words)
        
        for _ in range(num_aug):
            a = text
            
            # Synonym replacement
            if alpha > 0:
                n_sr = max(1, int(alpha * num_words))
                a = self.synonym_replacement(a, n_sr)
            
            # Random insertion
            if alpha > 0:
                n_ri = max(1, int(alpha * num_words))
                a = self.random_insertion(a, n_ri)
            
            # Random swap
            if alpha > 0:
                n_rs = max(1, int(alpha * num_words))
                a = self.random_swap(a, n_rs)
            
            # Random deletion
            if alpha > 0:
                a = self.random_deletion(a, alpha)
            
            augmented_texts.append(a)
        
        return augmented_texts
    
    def advanced_augmentation(self, text: str, method: str = 'contextual') -> str:
        """Apply advanced augmentation using pre-trained models"""
        try:
            if method == 'contextual' and self.contextual_aug:
                return self.contextual_aug.augment(text)[0]
            elif method == 'synonym' and self.synonym_aug:
                return self.synonym_aug.augment(text)[0]
            elif method == 'keyboard' and self.keyboard_aug:
                return self.keyboard_aug.augment(text)[0]
            elif method == 'random_char' and self.random_char_aug:
                return self.random_char_aug.augment(text)[0]
            else:
                # Fallback to EDA
                return self.easy_data_augmentation(text, alpha=0.1, num_aug=1)[0]
                
        except Exception as e:
            print(f"⚠️ Advanced augmentation failed: {e}")
            return self.easy_data_augmentation(text, alpha=0.1, num_aug=1)[0]
    
    def augment_text(self, text: str, methods: List[str] = None, num_aug: int = 2) -> List[str]:
        """Apply multiple augmentation methods"""
        if methods is None:
            methods = ['eda', 'contextual', 'synonym', 'back_translation']
        
        augmented_texts = []
        
        for method in methods[:num_aug]:
            try:
                if method == 'eda':
                    aug_texts = self.easy_data_augmentation(text, alpha=0.1, num_aug=1)
                    augmented_texts.extend(aug_texts)
                elif method == 'back_translation':
                    aug_text = self.back_translation(text)
                    if aug_text != text:  # Only add if actually changed
                        augmented_texts.append(aug_text)
                elif method in ['contextual', 'synonym', 'keyboard', 'random_char']:
                    aug_text = self.advanced_augmentation(text, method)
                    if aug_text != text:  # Only add if actually changed
                        augmented_texts.append(aug_text)
                else:
                    # Custom method
                    if method == 'random_swap':
                        augmented_texts.append(self.random_swap(text, 2))
                    elif method == 'random_deletion':
                        augmented_texts.append(self.random_deletion(text, 0.15))
                        
            except Exception as e:
                print(f"⚠️ Augmentation method {method} failed: {e}")
                continue
        
        # Remove duplicates and original text
        unique_augmented = []
        for aug_text in augmented_texts:
            if aug_text not in unique_augmented and aug_text != text:
                unique_augmented.append(aug_text)
        
        return unique_augmented

class HateSpeechDataset(Dataset):
    """PyTorch Dataset for hate speech classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class WeightedTrainer(Trainer):
    """Custom Trainer with class weighting for imbalanced datasets"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Move class weights to same device as logits
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get('loss')
        
        return (loss, outputs) if return_outputs else loss

class HateSpeechTransformerTrainer:
    """Transformer fine-tuning with text augmentation and balancing"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.categories = [category.value for category in HateSpeechCategory]
        self.label_to_id = {label: idx for idx, label in enumerate(self.categories)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize augmenter
        self.augmenter = TextAugmenter()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.categories),
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"🤖 Initialized {model_name} for {len(self.categories)} categories")
        print(f"📋 Categories: {', '.join(self.categories)}")
        print(f"🔧 Device: {self.device}")
    
    def get_balancing_strategy(self, strategy: str, random_state: int = 42):
        """Get balancing strategy from imbalanced-learn"""
        strategies = {
            'smote': SMOTE(random_state=random_state, k_neighbors=3),
            'adasyn': ADASYN(random_state=random_state, n_neighbors=3),
            'borderline_smote': BorderlineSMOTE(random_state=random_state, k_neighbors=3),
            'random_oversample': RandomOverSampler(random_state=random_state),
            'random_undersample': RandomUnderSampler(random_state=random_state),
            'smoteenn': SMOTEENN(random_state=random_state),
            'smotetomek': SMOTETomek(random_state=random_state),
            'none': None
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
        
        return strategies[strategy]
    
    def text_to_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Convert texts to numerical features for SMOTE"""
        if fit or not hasattr(self, 'vectorizer'):
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            features = self.vectorizer.fit_transform(texts).toarray()
        else:
            features = self.vectorizer.transform(texts).toarray()
        
        return features
    
    def apply_augmentation(self, texts: List[str], labels: List[int], 
                         aug_methods: List[str] = None, 
                         target_minority_size: int = None) -> Tuple[List[str], List[int]]:
        """Apply text augmentation to minority classes"""
        
        if not aug_methods or aug_methods == ['none']:
            return texts, labels
        
        print(f"🔄 Applying text augmentation with methods: {aug_methods}")
        
        # Analyze class distribution
        label_counts = Counter(labels)
        print(f"📊 Original distribution: {dict(label_counts)}")
        
        # Determine target size for minority classes
        if target_minority_size is None:
            # Use median as target size
            target_minority_size = int(np.median(list(label_counts.values())))
            target_minority_size = max(target_minority_size, 100)  # Minimum 50
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Apply augmentation to minority classes
        for label_id, count in label_counts.items():
            category_name = self.id_to_label[label_id]
            
            if count < target_minority_size:
                # Get texts for this category
                category_texts = [texts[i] for i, l in enumerate(labels) if l == label_id]
                needed_samples = target_minority_size - count
                
                print(f"🔄 Augmenting {category_name}: {count} → {target_minority_size} (+{needed_samples})")
                
                # Generate augmented samples
                generated_count = 0
                attempts = 0
                max_attempts = needed_samples * 3  # Avoid infinite loops
                
                while generated_count < needed_samples and attempts < max_attempts:
                    # Select random text from this category
                    source_text = random.choice(category_texts)
                    
                    # Apply random augmentation method
                    aug_method = random.choice(aug_methods)
                    
                    try:
                        if aug_method == 'eda':
                            aug_texts = self.augmenter.easy_data_augmentation(source_text, alpha=0.15, num_aug=1)
                        elif aug_method == 'contextual':
                            aug_texts = [self.augmenter.advanced_augmentation(source_text, 'contextual')]
                        elif aug_method == 'back_translation':
                            # Use different intermediate languages
                            lang = random.choice(['es', 'fr', 'de', 'it'])
                            aug_texts = [self.augmenter.back_translation(source_text, lang)]
                        elif aug_method == 'multi':
                            # Apply multiple methods
                            aug_texts = self.augmenter.augment_text(source_text, 
                                                                  ['eda', 'contextual', 'synonym'], 
                                                                  num_aug=1)
                        else:
                            # Fallback to EDA
                            aug_texts = self.augmenter.easy_data_augmentation(source_text, alpha=0.1, num_aug=1)
                        
                        # Add valid augmented texts
                        for aug_text in aug_texts:
                            if (aug_text != source_text and 
                                len(aug_text.strip()) > 5 and 
                                generated_count < needed_samples):
                                
                                augmented_texts.append(aug_text)
                                augmented_labels.append(label_id)
                                generated_count += 1
                                
                                if generated_count % 10 == 0:
                                    print(f"  Generated {generated_count}/{needed_samples} samples for {category_name}")
                    
                    except Exception as e:
                        print(f"⚠️ Augmentation failed for {category_name}: {e}")
                    
                    attempts += 1
                
                print(f"✅ Generated {generated_count} augmented samples for {category_name}")
        
        # Print final distribution
        final_counts = Counter(augmented_labels)
        print(f"📊 Augmented distribution: {dict(final_counts)}")
        print(f"✅ Total samples: {len(texts)} → {len(augmented_texts)}")
        
        return augmented_texts, augmented_labels
    
    def apply_balancing(self, texts: List[str], labels: List[int], 
                       strategy: str = 'smote') -> Tuple[List[str], List[int]]:
        """Apply balancing technique using imbalanced-learn"""
        
        if strategy == 'none':
            return texts, labels
        
        print(f"⚖️ Applying {strategy.upper()} balancing...")
        
        # Print original distribution
        original_dist = Counter(labels)
        print(f"📊 Pre-balancing distribution: {dict(original_dist)}")
        
        # Convert texts to features for SMOTE-based methods
        if strategy in ['smote', 'adasyn', 'borderline_smote', 'smoteenn', 'smotetomek']:
            X = self.text_to_features(texts, fit=True)
        else:
            # For simple over/under sampling, we can use indices
            X = np.arange(len(texts)).reshape(-1, 1)
        
        y = np.array(labels)
        
        # Get balancing strategy
        sampler = self.get_balancing_strategy(strategy)
        
        try:
            # Apply balancing
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # For SMOTE-based methods, we need to generate new texts
            if strategy in ['smote', 'adasyn', 'borderline_smote', 'smoteenn', 'smotetomek']:
                # For synthetic samples, we'll duplicate the closest original texts
                resampled_texts = []
                resampled_labels = []
                
                # Find original sample indices by matching features
                for i, (features, label) in enumerate(zip(X_resampled, y_resampled)):
                    # Find closest original sample
                    similarities = np.dot(X, features)
                    closest_idx = np.argmax(similarities)
                    resampled_texts.append(texts[closest_idx])
                    resampled_labels.append(int(label))
                
            else:
                # For simple sampling, use the indices directly
                indices = X_resampled.flatten()
                resampled_texts = [texts[i] for i in indices]
                resampled_labels = y_resampled.tolist()
            
            # Print new distribution
            new_dist = Counter(resampled_labels)
            print(f"📊 Post-balancing distribution: {dict(new_dist)}")
            print(f"✅ Samples: {len(texts)} → {len(resampled_texts)}")
            
            return resampled_texts, resampled_labels
            
        except Exception as e:
            print(f"⚠️ Balancing failed: {e}")
            print(f"📋 Falling back to random oversampling...")
            
            # Fallback to simple random oversampling
            sampler = RandomOverSampler(random_state=42)
            X_simple = np.arange(len(texts)).reshape(-1, 1)
            X_resampled, y_resampled = sampler.fit_resample(X_simple, y)
            
            indices = X_resampled.flatten()
            resampled_texts = [texts[i] for i in indices]
            resampled_labels = y_resampled.tolist()
            
            new_dist = Counter(resampled_labels)
            print(f"📊 Fallback distribution: {dict(new_dist)}")
            
            return resampled_texts, resampled_labels
    
    def load_and_prepare_data(self, data_files: List[str], text_column: str = "text", 
                            label_column: str = "category",
                            augmentation_methods: List[str] = None,
                            balance_strategy: str = 'smote') -> tuple:
        """Load and prepare data from multiple files with augmentation and balancing"""
        if isinstance(data_files, str):
            data_files = [data_files]  # Handle single file as list
        
        print(f"📚 Loading data from {len(data_files)} file(s): {data_files}")
        
        all_dfs = []
        for data_file in data_files:
            print(f"  📄 Loading {data_file}")
            df = pd.read_csv(data_file)
            
            if text_column not in df.columns or label_column not in df.columns:
                raise ValueError(f"Required columns '{text_column}' and '{label_column}' not found in {data_file}")
            
            # Add source file info
            df['source_file'] = data_file
            all_dfs.append(df)
            print(f"     └─ {len(df)} samples loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"📋 Combined dataset: {len(combined_df)} total samples")
        
        # Print distribution by source file
        print("📊 Distribution by source file:")
        for file in combined_df['source_file'].unique():
            file_count = len(combined_df[combined_df['source_file'] == file])
            print(f"  📄 {file}: {file_count} samples")
        
        # Filter valid categories
        valid_data = combined_df[combined_df[label_column].isin(self.categories)].copy()
        
        if len(valid_data) == 0:
            raise ValueError(f"No valid data found. Labels must be one of: {self.categories}")
        
        # Print category distribution
        category_dist = valid_data[label_column].value_counts()
        print(f"🎯 Category distribution after filtering:")
        for category, count in category_dist.items():
            print(f"  📌 {category}: {count} samples")
        
        # Prepare texts and labels
        texts = valid_data[text_column].astype(str).tolist()
        labels = [self.label_to_id[label] for label in valid_data[label_column]]
        
        # Apply augmentation first
        if augmentation_methods and augmentation_methods != ['none']:
            texts, labels = self.apply_augmentation(texts, labels, augmentation_methods)
        
        # Then apply balancing
        if balance_strategy != 'none':
            texts, labels = self.apply_balancing(texts, labels, balance_strategy)
        
        print(f"✅ Final dataset: {len(texts)} samples")
        return texts, labels
    
    def create_datasets(self, texts: List[str], labels: List[int], 
                       test_size: float = 0.2) -> tuple:
        """Split data and create PyTorch datasets"""
        print(f"🔀 Splitting data (test_size={test_size})")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = HateSpeechDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = HateSpeechDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        print(f"📚 Train samples: {len(train_dataset)}")
        print(f"🔍 Validation samples: {len(val_dataset)}")
        
        # Print class distribution in splits
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        print(f"🎯 Train distribution: {dict(train_dist)}")
        print(f"🎯 Val distribution: {dict(val_dist)}")
        
        return train_dataset, val_dataset, train_labels
    
    def compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Compute class weights for imbalanced dataset"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class accuracy
        per_class_acc = {}
        for i, category in enumerate(self.categories):
            mask = labels == i
            if mask.sum() > 0:
                per_class_acc[f"acc_{category}"] = accuracy_score(labels[mask], predictions[mask])
        
        return {"accuracy": accuracy, **per_class_acc}
    
    def train(self, train_dataset, val_dataset, train_labels, output_dir: str = "./hate_speech_model",
              num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              use_class_weights: bool = True):
        """Fine-tune the transformer model"""
        print(f"🚀 Starting training...")
        print(f"📁 Output directory: {output_dir}")
        print(f"⚙️ Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Compute class weights if requested
        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(train_labels)
            print(f"⚖️ Class weights: {class_weights.tolist()}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            seed=42,
            warmup_ratio=0.1,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
        )
        
        # Initialize trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        print("🔥 Training started...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training complete! Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, trainer, val_dataset) -> Dict:
        """Evaluate the trained model"""
        print("📊 Evaluating model...")
        
        # Get predictions
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Convert back to category names
        y_pred_labels = [self.id_to_label[pred] for pred in y_pred]
        y_true_labels = [self.id_to_label[true] for true in y_true]
        
        # Print classification report
        report = classification_report(y_true_labels, y_pred_labels, 
                                     target_names=self.categories, output_dict=True)
        
        print("\n📈 Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels, target_names=self.categories))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.categories)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.categories, yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report
    
    def predict(self, texts: List[str], model_path: str = None) -> List[str]:
        """Predict categories for new texts with device handling"""
        if model_path:
            # Load saved model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    truncation=True, 
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move inputs to same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = self.model(**inputs)
                predicted_id = torch.argmax(outputs.logits, dim=-1).item()
                predicted_category = self.id_to_label[predicted_id]
                predictions.append(predicted_category)
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer with text augmentation and balancing")
    parser.add_argument("--data", nargs='+', required=True, help="CSV file(s) with training data (can specify multiple files)")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--label-column", default="category", help="Name of label column")
    parser.add_argument("--model", default="distilbert-base-uncased", help="Pretrained model name")
    parser.add_argument("--output-dir", default="./hate_speech_model", help="Output directory")
    parser.add_argument("--balance-strategy", default="smote", 
                       choices=['smote', 'adasyn', 'borderline_smote', 'random_oversample', 
                               'random_undersample', 'smoteenn', 'smotetomek', 'none'],
                       help="Balancing strategy to use")
    parser.add_argument("--augmentation", nargs='+', default=['eda', 'contextual'],
                       choices=['eda', 'contextual', 'synonym', 'back_translation', 'multi', 'none'],
                       help="Text augmentation methods to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer_obj = HateSpeechTransformerTrainer(
            model_name=args.model,
            max_length=args.max_length
        )
        
        print(f"🎯 Using balancing strategy: {args.balance_strategy.upper()}")
        print(f"🔄 Using augmentation methods: {args.augmentation}")
        
        # Load and prepare data
        texts, labels = trainer_obj.load_and_prepare_data(
            args.data, args.text_column, args.label_column,
            augmentation_methods=args.augmentation,
            balance_strategy=args.balance_strategy
        )
        
        # Create datasets
        train_dataset, val_dataset, train_labels = trainer_obj.create_datasets(
            texts, labels, test_size=args.test_size
        )
        
        # Train model
        trainer = trainer_obj.train(
            train_dataset, val_dataset, train_labels,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_class_weights=not args.no_class_weights
        )
        
        # Evaluate model
        report = trainer_obj.evaluate(trainer, val_dataset)
        
        # Test with sample predictions
        print("\n🔮 Testing with sample predictions:")
        sample_texts = [
            "I hate all minorities and they should be deported",
            "Women are inferior to men and belong in the kitchen", 
            "This is just a normal conversation about the weather",
            "People with disabilities are worthless and should die",
            "Muslims are all terrorists and should be banned",
            "Gay people are disgusting and unnatural",
            "You look terrible today, so ugly",
            "Poor people are lazy and deserve nothing"
        ]
        
        predictions = trainer_obj.predict(sample_texts)
        for text, pred in zip(sample_texts, predictions):
            print(f"  Text: '{text}' → {pred}")
        
        print(f"\n🎉 Training complete! Model saved to {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 