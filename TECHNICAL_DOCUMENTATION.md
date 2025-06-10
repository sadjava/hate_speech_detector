# 📚 Technical Documentation: Multilingual Hate Speech Detection

## Abstract

This document presents a technical overview of our multilingual hate speech detection system, designed to classify text into 8 distinct categories across English and Serbian languages. The system employs a fine-tuned XLM-RoBERTa model with explainable AI features, achieving 87.3% accuracy.

---

## 1. 🔬 Methodology

### 1.1 Model Architecture

#### Base Model Selection
- **Model**: XLM-RoBERTa (Cross-lingual Language Model)
- **Parameters**: 560M parameters
- **Architecture**: 24 layers, 16 attention heads, 1024 hidden dimensions
- **Rationale**: Chosen for superior multilingual capabilities and proven performance on Serbian language tasks

#### Fine-tuning Architecture
```python
XLM-RoBERTa Base Model
├── Transformer Layers (24 layers)
├── Attention Heads (16 heads per layer)
├── Hidden Dimension: 1024
├── Classification Head
│   ├── Dropout Layer (p=0.1)
│   ├── Linear Layer (1024 → 512)
│   ├── ReLU Activation
│   ├── Dropout Layer (p=0.2)
│   └── Output Layer (512 → 8)
└── Softmax Activation
```

### 1.2 Training Configuration

#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 5 |
| Max Sequence Length | 512 |
| Warmup Steps | 100 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |


### 1.3 Data Balancing and Augmentation

#### Data Augmentation Methods
1. **AI Synthetic Generation**: 
   - Generated 300 English synthetic examples
   - Generated 200 Serbian synthetic examples
   - Improved dataset balance across categories
2. **SMOTE**: Synthetic Minority Oversampling Technique
3. **EDA (Easy Data Augmentation)**:
   - Synonym replacement (15% of words)
   - Random insertion (10% probability)
   - Random swap (10% probability)
   - Random deletion (10% probability)
4. **Back Translation**: Serbian ↔ English ↔ German ↔ Serbian

---

## 2. 📊 Dataset Description

### 2.1 Dataset Composition

#### Overall Statistics
| Language | Real Samples | Synthetic Samples | Total | Sources |
|----------|--------------|-------------------|-------|---------|
| English | ~700 | 300 | ~1,000 | 4chan, KiwiFarms + AI generation |
| Serbian | ~500 | 200 | ~700 | Teacher dataset + AI generation |
| **Total** | **~1,200** | **500** | **~1,700** | **Mixed collection** |

#### Category Distribution
The dataset covers 8 hate speech categories:
1. **Race**: Racial discrimination and slurs
2. **Sexual Orientation**: Homophobic content and LGBTQ+ discrimination  
3. **Gender**: Sexist language and gender-based harassment
4. **Physical Appearance**: Body shaming and appearance-based discrimination
5. **Religion**: Religious discrimination and slurs
6. **Class**: Economic discrimination and classist language
7. **Disability**: Ableist language and disability discrimination
8. **Appropriate**: Non-hateful, constructive content

### 2.2 Data Collection Process

#### English Dataset Collection
- **Sources**: 4chan and KiwiFarms forums
- **Collection Method**: Web scraping and parsing
- **Total Samples**: ~700 posts
- **Content**: User-generated forum discussions and comments
- **Filtering**: Removed duplicates and irrelevant content

#### Serbian Dataset Collection
- **Source**: Teacher-provided academic dataset
- **Total Samples**: ~500 samples
- **Content**: Pre-collected Serbian language hate speech examples
- **Quality**: Professionally curated academic dataset

### 2.3 Annotation Process

#### Hate Speech Categories

1. **Race**: Racial slurs, stereotypes, discrimination
2. **Sexual Orientation**: Homophobic language, LGBTQ+ discrimination
3. **Gender**: Sexist language, gender-based harassment
4. **Physical Appearance**: Body shaming, appearance-based discrimination
5. **Religion**: Religious slurs, Islamophobia, anti-Semitism
6. **Class**: Economic discrimination, classist language
7. **Disability**: Ableist language, disability discrimination
8. **Appropriate**: Non-hateful, constructive content

#### Annotation Process

**Phase 1: Manual Annotation**
- **Initial Labels**: 50 samples manually labeled by team
- **Categories**: 8 hate speech categories defined
- **Quality Control**: Careful review and consensus on definitions

**Phase 2: AI-Assisted Labeling**
- **Model Used**: GPT-4.1-nano for automated labeling
- **Method**: Few-shot learning with manually labeled examples
- **Coverage**: Remaining ~650 English samples
- **Prompt Engineering**: Category definitions and examples provided

**Phase 3: Semi-Supervised Review**
- **Validation**: Manual review of AI-generated labels
- **Corrections**: Fixed inconsistent or incorrect labels
- **Quality Assurance**: Ensured consistency across dataset

**Phase 4: Synthetic Data Generation**
- **AI Generation**: Created additional synthetic examples using AI models
- **English Synthetic**: 300 additional samples generated
- **Serbian Synthetic**: 200 additional samples generated
- **Purpose**: Data augmentation to improve model robustness
- **Quality Control**: Generated examples reviewed for realism and accuracy

**Final Dataset**: Combined real samples (~1,200) + synthetic samples (500) = ~1,700 total

---

## 3. 🧪 Experimental Setup

### 3.1 Infrastructure

#### Hardware
- **GPU**: NVIDIA RTX 3060 (6GB VRAM)
- **Platform**: Laptop configuration
- **Memory Constraints**: Limited by 6GB GPU memory

#### Software
- **Framework**: PyTorch 2.4.0
- **Transformers**: 4.30.2
- **CUDA**: 12.4
- **Python**: 3.12.9

### 3.2 Evaluation Setup

#### Data Split
- **Training**: 70% (~1,190 samples)
- **Validation**: 15% (~255 samples)
- **Test**: 15% (~255 samples)
- **Total**: ~1,700 samples (Real + Synthetic, English + Serbian)
- **Stratification**: Maintains category distribution across languages and data types

#### Metrics
- **Primary**: F1-score (macro-averaged)
- **Secondary**: Precision, Recall, Accuracy
- **Cross-validation**: 5-fold stratified

---

## 4. 📈 Results

### 4.1 Overall Performance

#### Main Results
| Metric | Score | 95% CI |
|--------|-------|---------|
| **Accuracy** | **87.3%** | ±2.1% |
| **F1-Score (Macro)** | **85.7%** | ±2.4% |
| **F1-Score (Weighted)** | **87.1%** | ±2.0% |
| **Precision (Macro)** | **86.2%** | ±2.3% |
| **Recall (Macro)** | **85.4%** | ±2.5% |

### 4.2 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix shows strong performance across all categories, with particularly high accuracy for Religion (91%) and appropriate content (90%).

### 4.3 Per-Category Performance

The model demonstrates strong performance across all 8 categories, with particularly high accuracy for religious hate speech detection and appropriate content classification. Performance metrics vary by category complexity and representation in the training data.

### 4.4 Cross-Lingual Performance

The model was trained on the combined English-Serbian dataset, enabling effective hate speech detection in both languages. The XLM-RoBERTa architecture provides strong cross-lingual transfer capabilities, allowing the model to leverage patterns learned from both language datasets.

---

## 5. 💡 Innovation Features

### 5.1 Contextual Analysis

#### Attention Visualization
The model implements attention visualization to provide interpretable results, highlighting which words or phrases contribute most to the hate speech classification decision.

#### Example Analysis
```
Text: "You people are all the same, causing problems"
Attention Weights: [You: 0.15, people: 0.89, are: 0.12, all: 0.78, same: 0.85, ...]
Prediction: Race (94.2% confidence)
```

---

## 6. 🚨 Limitations and Future Work

### 6.1 Current Limitations

1. **Language Coverage**: Limited to English and Serbian
2. **Context Length**: 512 token limit may truncate long texts
3. **Cultural Bias**: Training data may reflect cultural biases
4. **Real-time Performance**: 2-second inference time on CPU
5. **Domain Adaptation**: Performance varies across platforms

### 6.2 Future Directions

1. **Multi-language Expansion**: Add Croatian, Bosnian, Montenegrin
2. **Real-time Optimization**: Model compression and quantization
3. **Bias Mitigation**: Advanced debiasing techniques
4. **Dynamic Learning**: Continuous learning from user feedback
5. **Multimodal Integration**: Text + image hate speech detection

---

## 7. 📋 Conclusion

### 7.1 Key Achievements

1. **High Accuracy**: 87.3% accuracy across 8 hate speech categories
2. **Multilingual Support**: First English-Serbian hate speech detection system
3. **Explainable AI**: Attention visualization for model interpretability
4. **Data Innovation**: Combined real and AI-generated synthetic training data

### 7.2 Impact

- **Research**: Advancing multilingual hate speech detection
- **Social**: Safer online communities for Serbian speakers
- **Technical**: Explainable AI in content moderation
- **Educational**: Open-source system for research and learning

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Dzhavid Sadreddinov, Dmitriy Dydalin, Amir Bikineyev  
**GitHub**: [hate_speech_detector](https://github.com/sadjava/hate_speech_detector) 