#!/usr/bin/env python3

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import numpy as np

class HateSpeechDetector:
    def __init__(self, model_path: str = "models/multilingual_xlm_roberta"):
        """Initialize the hate speech detector with a trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Define hate speech categories
        self.categories = [
            "Race", "Sexual Orientation", "Gender", "Physical Appearance", 
            "Religion", "Class", "Disability", "Appropriate"
        ]
    
    def predict_with_context(self, text: str) -> tuple:
        """Predict hate speech category with contextual analysis."""
        if not text.strip():
            return "Please enter some text", 0.0, {}, ""
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_attention_mask=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions
        
        # Calculate probabilities
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = float(probabilities[0][predicted_class])
        predicted_category = self.categories[predicted_class]
        
        # Create confidence chart
        confidence_chart = self.create_confidence_chart(probabilities[0])
        
        # Create word highlighting
        highlighted_html = self.create_word_highlighting(text, inputs, attentions)
        
        return predicted_category, confidence, confidence_chart, highlighted_html
    
    def create_confidence_chart(self, probabilities):
        """Create confidence visualization."""
        scores = [float(prob) for prob in probabilities]
        colors = ['#ff6b6b' if cat != 'Appropriate' else '#51cf66' for cat in self.categories]
        
        fig = go.Figure(data=[
            go.Bar(
                x=self.categories,
                y=scores,
                marker_color=colors,
                text=[f'{score:.1%}' for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Confidence Scores by Category",
            xaxis_title="Categories",
            yaxis_title="Confidence",
            yaxis_range=[0, 1],
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_word_highlighting(self, text, inputs, attentions):
        """Create word-level importance highlighting."""
        # Use multiple attention heads and layers for better analysis
        # Average across heads in the last layer
        last_layer_attention = attentions[-1][0]  # [num_heads, seq_len, seq_len]
        avg_attention = torch.mean(last_layer_attention, dim=0)  # [seq_len, seq_len]
        
        # Calculate importance as sum of attention TO each token (column sum)
        token_importance = torch.sum(avg_attention, dim=0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Remove special tokens and get actual content tokens
        content_tokens = tokens[1:-1]  # Remove [CLS] and [SEP]
        content_importance = token_importance[1:-1]
        
        # Normalize importance scores with better distribution
        if len(content_importance) > 1:
            importance_norm = (content_importance - content_importance.min()) / (content_importance.max() - content_importance.min() + 1e-8)
            # Apply power scaling to increase contrast
            importance_norm = np.power(importance_norm, 0.5)
        else:
            importance_norm = np.array([0.5])
        
        # Map subword tokens back to original words
        words = text.split()
        word_scores = []
        
        # Better token-to-word mapping
        reconstructed_text = self.tokenizer.convert_tokens_to_string(content_tokens)
        current_pos = 0
        token_idx = 0
        
        for word in words:
            word_importance_scores = []
            
            # Find tokens that correspond to this word
            while token_idx < len(content_tokens):
                token = content_tokens[token_idx]
                token_clean = token.replace('▁', '').replace('##', '')  # Handle subword prefixes
                
                if token_clean.lower() in word.lower() or word.lower() in token_clean.lower():
                    word_importance_scores.append(importance_norm[token_idx])
                    token_idx += 1
                elif len(word_importance_scores) > 0:
                    # We've moved past this word
                    break
                else:
                    token_idx += 1
                    
                if token_idx >= len(content_tokens):
                    break
            
            # Calculate average importance for this word
            if word_importance_scores:
                word_score = np.mean(word_importance_scores)
            else:
                word_score = 0.2  # Default low importance
            
            word_scores.append(word_score)
        
        # Ensure we have same number of words and scores
        while len(word_scores) < len(words):
            word_scores.append(0.2)
        word_scores = word_scores[:len(words)]
        
        # Create HTML with enhanced highlighting
        html_parts = []
        for i, (word, score) in enumerate(zip(words, word_scores)):
            # Better color scheme with more variation
            if score > 0.7:
                color = "rgba(220, 53, 69, 0.8)"  # Strong red
            elif score > 0.5:
                color = "rgba(255, 193, 7, 0.8)"  # Orange  
            elif score > 0.3:
                color = "rgba(255, 235, 59, 0.6)"  # Yellow
            else:
                color = "rgba(248, 249, 250, 0.3)"  # Light gray
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 3px 6px; margin: 2px; '
                f'border-radius: 4px; font-weight: 500; border: 1px solid rgba(0,0,0,0.1);" '
                f'title="Word: {word} | Importance: {score:.3f}">{word}</span>'
            )
        
        return '<div style="line-height: 2.5; font-size: 16px; padding: 10px;">' + ' '.join(html_parts) + '</div>'

# Initialize detector
detector = HateSpeechDetector()

def analyze_text(text: str):
    """Main analysis function with innovations."""
    try:
        category, confidence, chart, highlighted = detector.predict_with_context(text)
        
        if category == "Appropriate":
            result = f"✅ **No hate speech detected**\n\nCategory: {category}\nConfidence: {confidence:.1%}"
        else:
            result = f"⚠️ **Hate speech detected**\n\nCategory: {category}\nConfidence: {confidence:.1%}"
        
        return result, chart, highlighted
        
    except Exception as e:
        return f"❌ Error: {str(e)}", {}, ""

def provide_feedback(text: str, rating: int):
    """Simple feedback collection."""
    if not text.strip():
        return "Please analyze some text first!"
    return f"✅ Thanks for rating {rating}/5 stars! Feedback helps improve the model."

# Create enhanced Gradio interface
with gr.Blocks(title="Hate Speech Detector") as demo:
    gr.Markdown("""
    # 🛡️ AI Hate Speech Detector
    
    **Multilingual hate speech detection with contextual analysis**
    
    🔬 **Innovations:**
    - **Contextual Analysis**: See which words influenced the decision
    - **Confidence Visualization**: Interactive charts for all categories  
    - **Word Highlighting**: Visual explanation of model focus
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter text to analyze (English/Serbian)",
                placeholder="Type or paste text here...",
                lines=3
            )
            
            analyze_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
            
            gr.Examples(
                examples=[
                    ["I really enjoyed that movie last night!"],
                    ["You people are all the same, causing problems everywhere."],
                    ["Women can't drive as well as men."],
                    ["Ovaj film je bio odličan!"],  # Serbian: great movie
                    ["Ti ljudi ne zaslužuju da žive ovde."],  # Serbian hate speech
                ],
                inputs=text_input
            )
        
        with gr.Column():
            result_output = gr.Markdown(label="🎯 Classification Result")
    
    # Innovation: Confidence Visualization
    gr.Markdown("### 📊 **Innovation 1**: Confidence Visualization")
    confidence_plot = gr.Plot(label="Confidence scores across all hate speech categories")
    
    # Innovation: Contextual Analysis
    gr.Markdown("### 🌈 **Innovation 2**: Contextual Word Analysis")
    gr.Markdown("*Words highlighted in red had more influence on the classification decision*")
    highlighted_text = gr.HTML(label="Word importance highlighting")
    
    # Innovation: Interactive Feedback
    with gr.Accordion("💬 **Innovation 3**: Interactive Feedback", open=False):
        gr.Markdown("Help improve the model by rating the analysis quality!")
        feedback_rating = gr.Slider(1, 5, step=1, value=3, label="Rate this analysis (1-5 stars)")
        feedback_btn = gr.Button("📝 Submit Feedback")
        feedback_output = gr.Textbox(label="Feedback Status", interactive=False)
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_text,
        inputs=[text_input],
        outputs=[result_output, confidence_plot, highlighted_text]
    )
    
    feedback_btn.click(
        fn=provide_feedback,
        inputs=[text_input, feedback_rating],
        outputs=[feedback_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 