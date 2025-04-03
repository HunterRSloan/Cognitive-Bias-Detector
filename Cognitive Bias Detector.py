import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Download all required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('omw-1.4')  # Open Multilingual Wordnet

# Expanded dataset with more balanced examples and augmented data
data = {
    'text': [
        # Original Confirmation Bias Examples
        "This confirms what we've always known about their policies",
        "New study supports existing views on climate change",
        "The evidence clearly backs up our original hypothesis",
        "I only read news sources that align with my beliefs",
        "These statistics prove what I've been saying all along",
        "Finally, research that validates our perspective",
        # Additional Confirmation Bias Examples
        "I prefer media that reinforces my existing beliefs",
        "The data perfectly matches what we expected to find",
        
        # Original Framing Bias Examples
        "Crime rates soar under new administration",
        "Economic growth reaches record highs",
        "The glass is half empty",
        "The glass is half full",
        "90% chance of survival with this treatment",
        "10% risk of death with this treatment",
        # Additional Framing Examples
        "Unemployment drops to historic low of 4%",
        "96% of workers still employed during crisis",
        
        # Original Anchoring Bias Examples
        "The initial price was $1000, but now it's only $800",
        "Starting salary is $50,000, which is a great deal",
        "The first offer sets the baseline for negotiations",
        "Original price $200, now reduced to $150",
        "Compare to high-end model at $2000",
        "Market value started at $1M, now only $800K",
        # Additional Anchoring Examples
        "Previous model cost $2000, this one is just $1500",
        "Compared to last year's budget of $1M, this is reasonable",
        
        # Original Availability Bias Examples
        "Recent plane crashes make me afraid to fly",
        "After seeing shark attacks on TV, I'm scared of swimming",
        "Hearing about lottery winners makes me think I could win",
        "News about burglaries makes me double-check my locks",
        "Stories about identity theft make me paranoid online",
        "Recent food poisoning cases make me avoid that restaurant",
        # Additional Availability Examples
        "After watching crime shows, I'm scared to walk alone",
        "Reading about successful startups makes me want to quit my job",
        
        # Original Dunning-Kruger Effect Examples
        "I've read one book on psychology, so I'm basically an expert",
        "After watching a few YouTube videos, I can fix any car",
        "I know everything there is to know about this topic",
        "How hard could brain surgery be? I've watched medical shows",
        "I could easily run a restaurant after watching cooking shows",
        "One coding tutorial and I'm ready to build the next Facebook",
        # Additional Dunning-Kruger Examples
        "Just learned Python yesterday, ready to build AI systems",
        "Read an article about quantum physics, now I can explain string theory",
        
        # Original Hindsight Bias Examples
        "I knew this would happen all along",
        "The signs were obvious from the beginning",
        "We should have seen the market crash coming",
        "It was clear from the start they would break up",
        "The outcome was predictable from day one",
        "Looking back, the warning signs were everywhere",
        # Additional Hindsight Examples
        "Anyone could have predicted this result",
        "In retrospect, the pattern was clear",
        
        # Original Neutral Examples
        "Studies show both positive and negative effects",
        "The research presents balanced findings",
        "The data suggests multiple possible outcomes",
        "Further investigation is needed to draw conclusions",
        "Evidence supports various interpretations",
        "Results are inconclusive and require more study",
        # Additional Neutral Examples
        "More research is needed to confirm these findings",
        "The evidence is currently insufficient to draw conclusions"
    ],
    'bias_type': [
        # Original + Additional Confirmation Bias
        'confirmation_bias', 'confirmation_bias', 'confirmation_bias',
        'confirmation_bias', 'confirmation_bias', 'confirmation_bias',
        'confirmation_bias', 'confirmation_bias',
        
        # Original + Additional Framing
        'framing', 'framing', 'framing', 'framing', 'framing', 'framing',
        'framing', 'framing',
        
        # Original + Additional Anchoring Bias
        'anchoring_bias', 'anchoring_bias', 'anchoring_bias',
        'anchoring_bias', 'anchoring_bias', 'anchoring_bias',
        'anchoring_bias', 'anchoring_bias',
        
        # Original + Additional Availability Bias
        'availability_bias', 'availability_bias', 'availability_bias',
        'availability_bias', 'availability_bias', 'availability_bias',
        'availability_bias', 'availability_bias',
        
        # Original + Additional Dunning-Kruger
        'dunning_kruger', 'dunning_kruger', 'dunning_kruger',
        'dunning_kruger', 'dunning_kruger', 'dunning_kruger',
        'dunning_kruger', 'dunning_kruger',
        
        # Original + Additional Hindsight Bias
        'hindsight_bias', 'hindsight_bias', 'hindsight_bias',
        'hindsight_bias', 'hindsight_bias', 'hindsight_bias',
        'hindsight_bias', 'hindsight_bias',
        
        # Original + Additional Neutral
        'none', 'none', 'none', 'none', 'none', 'none',
        'none', 'none'
    ]
}

class CognitiveBiasDetector:
    def __init__(self):
        # Create multiple vectorizers with different parameters
        self.vectorizer1 = TfidfVectorizer(
            max_features=2000,  # Keep this reduced
            ngram_range=(1, 2),
            min_df=2,          # Reduced back to allow more terms
            max_df=0.95,       # Increased to allow more common terms
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        
        self.vectorizer2 = TfidfVectorizer(
            max_features=1500,  # Keep this reduced
            ngram_range=(2, 3),
            min_df=2,          # Reduced back
            max_df=0.95,       # Increased
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        
        self.model = RandomForestClassifier(
            n_estimators=3500,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight={
                0: 2.5,  # anchoring_bias
                1: 4.0,  # availability_bias (increased further)
                2: 3.0,  # confirmation_bias
                3: 4.0,  # dunning_kruger (increased further)
                4: 2.0,  # framing
                5: 3.5,  # hindsight_bias
                6: 2.0   # none (increased to improve neutral detection)
            },
            n_jobs=-1,
            criterion='entropy',
            bootstrap=True,
            max_features='sqrt',
            oob_score=True
        )
        
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with specific bias patterns"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace numbers and currencies with semantic tokens
        text = re.sub(r'\d+%', ' percentage_value ', text)
        text = re.sub(r'\$\d+[km]?', ' price_value ', text)
        text = re.sub(r'\d+k', ' thousand_value ', text)
        text = re.sub(r'\d+m', ' million_value ', text)
        
        # Enhanced pattern markers
        # Confirmation bias patterns (expanded)
        text = re.sub(r'(confirms?|supports?|validates?|proves?|back(s|ed) up|align(s|ed)?)', ' confirmation_marker ', text)
        text = re.sub(r'(always|already|consistently) (known|believed|thought|said|claimed)', ' prior_belief_marker ', text)
        text = re.sub(r'(only|exclusively) (read|trust|believe|follow)', ' selective_exposure_marker ', text)
        
        # Framing bias patterns (expanded)
        text = re.sub(r'(soars?|plummets?|crashes?|dramatic(ally)?|surge|plunge)', ' extreme_change_marker ', text)
        text = re.sub(r'(half empty|half full|positive|negative|good|bad) (side|aspect|view|outlook)', ' framing_perspective_marker ', text)
        text = re.sub(r'(chance|risk|probability) of (survival|death|success|failure)', ' outcome_framing_marker ', text)
        
        # Anchoring bias patterns (expanded)
        text = re.sub(r'(initial|original|starting|first|baseline) (price|value|offer|point)', ' anchor_point_marker ', text)
        text = re.sub(r'(now only|reduced to|special price|compared to|reference)', ' price_reduction_marker ', text)
        text = re.sub(r'(sets?|establish(es)?|determine(s)?) (baseline|standard|benchmark)', ' anchoring_effect_marker ', text)
        
        # Availability bias patterns (expanded)
        text = re.sub(r'(after seeing|recent|lately|in the news|media|headlines|hearing about)', ' recency_marker ', text)
        text = re.sub(r'(afraid|scared|worried|paranoid|anxious|concerned|fear) (of|about|that)', ' fear_response_marker ', text)
        text = re.sub(r'(makes? me|causing|leads? to|got me) (think|believe|feel|worry|concerned)', ' availability_influence_marker ', text)
        
        # Dunning-Kruger patterns (expanded)
        text = re.sub(r'(basically|easily|just|simply|obviously|already|now) (an expert|understand|know|master|explain|teach|build)', ' overconfidence_marker ', text)
        text = re.sub(r'(one|few|couple|brief|just|recently|yesterday) (book|video|tutorial|course|lesson|article|day)', ' limited_exposure_marker ', text)
        text = re.sub(r'(how hard could|cant be that|easy to|simple enough|ready to|qualified to)', ' task_simplification_marker ', text)
        
        # Hindsight bias patterns (expanded)
        text = re.sub(r'(knew|obvious|clear|predictable|saw) .*(all along|from the (start|beginning)|coming)', ' hindsight_marker ', text)
        text = re.sub(r'(should have (known|seen|predicted)|looking back|in retrospect)', ' retrospect_marker ', text)
        text = re.sub(r'(saw this|expected|anticipated|could have predicted) .*(coming|happening|occurring)', ' prediction_claim_marker ', text)
        
        # Add special tokens for common bias patterns
        text = re.sub(r'(always|never|every|all|none|impossible|definitely)', r'absolute_term_\1', text)
        text = re.sub(r'(clearly|obviously|certainly|undoubtedly|absolutely)', r'certainty_term_\1', text)
        text = re.sub(r'(but now|only|just|merely|simply)', r'contrast_term_\1', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Enhanced important words list with bias-specific terms
        important_words = {
            # General modifiers
            'no', 'not', 'none', 'nothing', 'never', 'only', 'but', 'however',
            'although', 'despite', 'always', 'every', 'all', 'clearly',
            'obviously', 'certainly', 'must', 'should', 'would', 'could',
            'might', 'may', 'perhaps', 'possibly', 'probably',
            
            # Bias-specific words
            'confirms', 'proves', 'supports', 'validates', 'believe',
            'think', 'know', 'understand', 'expert', 'easily',
            'obvious', 'clear', 'predictable', 'knew', 'saw',
            'afraid', 'scared', 'worried', 'recent', 'news',
            'price', 'value', 'offer', 'deal', 'special',
            'dramatic', 'soar', 'plummet', 'crash', 'increase',
            'decrease', 'rise', 'fall', 'more', 'less'
        }
        
        # Remove stopwords and lemmatize, but keep important words
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if (token not in self.stop_words or token in important_words) and len(token) > 1
        ]
        
        return ' '.join(tokens)

    def extract_features(self, texts):
        """Combine features from multiple vectorizers with additional pattern features"""
        # Get TF-IDF features
        X1 = self.vectorizer1.transform(texts)
        X2 = self.vectorizer2.transform(texts)
        
        # Create pattern-based features with confidence scores
        pattern_features = []
        for text in texts:
            # Basic pattern presence
            features = {
                'has_confirmation_marker': 1 if 'confirmation_marker' in text else 0,
                'has_prior_belief_marker': 1 if 'prior_belief_marker' in text else 0,
                'has_extreme_change_marker': 1 if 'extreme_change_marker' in text else 0,
                'has_framing_perspective_marker': 1 if 'framing_perspective_marker' in text else 0,
                'has_anchor_point_marker': 1 if 'anchor_point_marker' in text else 0,
                'has_price_reduction_marker': 1 if 'price_reduction_marker' in text else 0,
                'has_recency_marker': 1 if 'recency_marker' in text else 0,
                'has_fear_marker': 1 if 'fear_response_marker' in text else 0,
                'has_overconfidence_marker': 1 if 'overconfidence_marker' in text else 0,
                'has_limited_exposure_marker': 1 if 'limited_exposure_marker' in text else 0,
                'has_hindsight_marker': 1 if 'hindsight_marker' in text else 0,
                'has_retrospect_marker': 1 if 'retrospect_marker' in text else 0,
                
                # New combined pattern features
                'has_availability_pattern': 1 if any(marker in text for marker in 
                    ['recency_marker', 'fear_response_marker', 'availability_influence_marker']) else 0,
                'has_dunning_kruger_pattern': 1 if any(marker in text for marker in 
                    ['overconfidence_marker', 'limited_exposure_marker', 'task_simplification_marker']) else 0,
                'has_framing_pattern': 1 if any(marker in text for marker in 
                    ['extreme_change_marker', 'framing_perspective_marker', 'outcome_framing_marker']) else 0,
                
                # Pattern combinations
                'confirmation_with_certainty': 1 if 'confirmation_marker' in text and 'certainty_term' in text else 0,
                'availability_with_fear': 1 if 'recency_marker' in text and 'fear_response_marker' in text else 0,
                'dunning_kruger_with_confidence': 1 if 'limited_exposure_marker' in text and 'overconfidence_marker' in text else 0,
                'framing_with_extreme': 1 if 'framing_perspective_marker' in text and 'extreme_change_marker' in text else 0,
                
                # Neutral indicators
                'has_balanced_perspective': 1 if 'both' in text or 'multiple' in text or 'various' in text else 0,
                'has_uncertainty_marker': 1 if any(word in text for word in ['possibly', 'perhaps', 'maybe', 'might']) else 0
            }
            pattern_features.append(list(features.values()))
        
        pattern_features = np.array(pattern_features)
        
        # Combine all features
        return np.hstack([X1.toarray(), X2.toarray(), pattern_features])

    def train(self, texts, y):
        """Train the model with combined features and enhanced validation"""
        # Fit vectorizers
        self.vectorizer1.fit(texts)
        self.vectorizer2.fit(texts)
        
        # Extract combined features
        X = self.extract_features(texts)
        
        # Perform stratified k-fold cross-validation
        cv_scores = cross_val_score(
            self.model, 
            X, 
            y, 
            cv=5, 
            scoring='f1_macro'
        )
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        self.model.fit(X, y)
        print(f"\nOut-of-bag score: {self.model.oob_score_:.3f}")
        
        # Get feature names and importance scores
        feature_names = (
            list(self.vectorizer1.get_feature_names_out()) +
            list(self.vectorizer2.get_feature_names_out()) +
            [
                'has_confirmation_marker',
                'has_prior_belief_marker',
                'has_extreme_change_marker',
                'has_framing_perspective_marker',
                'has_anchor_point_marker',
                'has_price_reduction_marker',
                'has_recency_marker',
                'has_fear_marker',
                'has_overconfidence_marker',
                'has_limited_exposure_marker',
                'has_hindsight_marker',
                'has_retrospect_marker',
                'has_availability_pattern',
                'has_dunning_kruger_pattern',
                'has_framing_pattern',
                'confirmation_with_certainty',
                'availability_with_fear',
                'dunning_kruger_with_confidence',
                'framing_with_extreme',
                'has_balanced_perspective',
                'has_uncertainty_marker'
            ]
        )
        
        # Print feature importance with bias type associations
        importances = self.model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        
        print("\nTop features by bias type:")
        bias_types = ['confirmation_bias', 'anchoring_bias', 'availability_bias', 
                     'dunning_kruger', 'framing', 'hindsight_bias', 'none']
        
        for bias_type in bias_types:
            relevant_features = [f for f, _ in sorted_features[:20] 
                               if any(term in f.lower() for term in bias_type.lower().split('_'))]
            if relevant_features:
                print(f"\n{bias_type.replace('_', ' ').title()} indicators:")
                for feature in relevant_features[:3]:
                    importance = dict(feature_importance)[feature]
                    print(f"  {feature}: {importance:.4f}")

    def evaluate(self, texts, y_test):
        """Evaluate the model with combined features"""
        X_test = self.extract_features(texts)
        y_pred = self.model.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                 target_names=self.label_encoder.classes_,
                                 zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def save_model(self, path: str) -> None:
        """Save the trained model and vectorizers"""
        joblib.dump({
            'model': self.model,
            'vectorizer1': self.vectorizer1,
            'vectorizer2': self.vectorizer2,
            'label_encoder': self.label_encoder
        }, path)

    def load_model(self, path: str) -> None:
        """Load a trained model and vectorizers"""
        saved_model = joblib.load(path)
        self.model = saved_model['model']
        self.vectorizer1 = saved_model['vectorizer1']
        self.vectorizer2 = saved_model['vectorizer2']
        self.label_encoder = saved_model['label_encoder']

    def predict_bias(self, text: str) -> Tuple[str, float]:
        processed = self.preprocess_text(text)
        features = self.extract_features([processed])
        proba = self.model.predict_proba(features)
        
        sorted_idx = np.argsort(proba[0])[::-1]
        sorted_classes = self.label_encoder.inverse_transform(sorted_idx)
        sorted_probs = proba[0][sorted_idx]
        
        # Adjusted base thresholds
        base_thresholds = {
            'confirmation_bias': 0.25,
            'anchoring_bias': 0.25,
            'availability_bias': 0.15,  # Further reduced
            'dunning_kruger': 0.18,    # Further reduced
            'framing': 0.28,
            'hindsight_bias': 0.22,
            'none': 0.40               # Increased for more selective neutral classification
        }
        
        # Enhanced pattern weights
        pattern_weights = {
            # Confirmation bias markers
            'confirmation_marker': 0.25,
            'prior_belief_marker': 0.20,
            'selective_exposure_marker': 0.20,
            
            # Availability bias markers
            'recency_marker': 0.35,     # Further increased
            'fear_response_marker': 0.35, # Further increased
            'availability_influence_marker': 0.30,
            
            # Dunning-Kruger markers
            'overconfidence_marker': 0.35, # Further increased
            'limited_exposure_marker': 0.30,
            'task_simplification_marker': 0.30,
            
            # Other markers unchanged
            'extreme_change_marker': 0.25,
            'framing_perspective_marker': 0.25,
            'outcome_framing_marker': 0.20,
            'hindsight_marker': 0.30,
            'retrospect_marker': 0.25,
            'prediction_claim_marker': 0.25,
            'anchor_point_marker': 0.25,
            'price_reduction_marker': 0.25,
            'anchoring_effect_marker': 0.20
        }
        
        # Simplified pattern scoring with error handling
        def get_pattern_score(text: str, markers: List[str], context_words: List[str]) -> float:
            try:
                base_score = sum(pattern_weights.get(m, 0.0) for m in markers if m in text)
                context_bonus = 0.1 if any(word in text for word in context_words) else 0
                return min(0.5, base_score + context_bonus)  # Cap maximum score
            except Exception as e:
                print(f"Warning: Error in pattern scoring - {e}")
                return 0.0
        
        pattern_scores = {
            'confirmation_bias': get_pattern_score(
                processed,
                ['confirmation_marker', 'prior_belief_marker', 'selective_exposure_marker'],
                ['believe', 'proves', 'validates']
            ),
            'availability_bias': get_pattern_score(
                processed,
                ['recency_marker', 'fear_response_marker', 'availability_influence_marker'],
                ['scared', 'afraid', 'worried']
            ),
            'dunning_kruger': get_pattern_score(
                processed,
                ['overconfidence_marker', 'limited_exposure_marker', 'task_simplification_marker'],
                ['expert', 'easily', 'simple']
            ),
            'framing': get_pattern_score(
                processed,
                ['extreme_change_marker', 'framing_perspective_marker', 'outcome_framing_marker'],
                ['dramatic', 'soar', 'plummet']
            ),
            'hindsight_bias': get_pattern_score(
                processed,
                ['hindsight_marker', 'retrospect_marker', 'prediction_claim_marker'],
                ['obvious', 'knew', 'predictable']
            ),
            'anchoring_bias': get_pattern_score(
                processed,
                ['anchor_point_marker', 'price_reduction_marker', 'anchoring_effect_marker'],
                ['price', 'value', 'offer', 'compared']
            ),
            'none': 0.0  # Default score for neutral class
        }
        
        # Get top prediction and confidence
        top_class = sorted_classes[0]
        confidence = sorted_probs[0]
        
        # Enhanced neutral detection with more indicators
        neutral_indicators = [
            'both' in processed,
            'multiple' in processed,
            'various' in processed,
            'further' in processed,
            'research' in processed,
            'study' in processed,
            'evidence' in processed,
            'data' in processed,
            'suggests' in processed,
            'may' in processed,
            'investigation' in processed,
            'inconclusive' in processed,
            'balanced' in processed,
            'unclear' in processed
        ]
        
        # Improved neutral check with weighted scoring
        neutral_score = sum(neutral_indicators) / len(neutral_indicators)
        max_pattern_score = max(pattern_scores.values())
        
        # Enhanced pattern-based decision making
        if top_class in pattern_scores:
            pattern_confidence = pattern_scores[top_class]
            
            # Special handling for specific bias types with lower thresholds
            if top_class == 'availability_bias' and pattern_confidence > 0.15:
                return top_class, max(confidence, 0.35), sorted_classes[:2], sorted_probs[:2]
            
            if top_class == 'dunning_kruger' and pattern_confidence > 0.15:
                return top_class, max(confidence, 0.35), sorted_classes[:2], sorted_probs[:2]
            
            if top_class == 'hindsight_bias' and pattern_confidence > 0.20:
                return top_class, max(confidence, 0.35), sorted_classes[:2], sorted_probs[:2]
            
            # Enhanced neutral detection
            if top_class == 'none':
                if neutral_score >= 0.35 and max_pattern_score < 0.20:  # More stringent requirements
                    return 'none', max(0.40, confidence), sorted_classes[:2], sorted_probs[:2]
                elif any(pattern_scores[bias] > 0.25 for bias in pattern_scores if bias != 'none'):
                    # If strong bias patterns exist, avoid neutral classification
                    next_best = next((cls for cls in sorted_classes if cls != 'none'), None)
                    if next_best:
                        return next_best, sorted_probs[sorted_classes == next_best][0], sorted_classes[:2], sorted_probs[:2]
            
            # Relaxed pattern match threshold with type-specific adjustments
            if pattern_confidence > 0.20 and confidence >= base_thresholds[top_class]:
                return top_class, confidence, sorted_classes[:2], sorted_probs[:2]
            
            # Check for better pattern matches with type-specific margins
            for cls, prob in zip(sorted_classes[1:], sorted_probs[1:]):
                if cls in pattern_scores:
                    margin = 1.25 if cls in ['availability_bias', 'dunning_kruger'] else 1.1
                    if pattern_scores[cls] > pattern_confidence * margin:
                        if prob > base_thresholds.get(cls, 0.25):
                            return cls, prob, sorted_classes[:2], sorted_probs[:2]
        
        # More stringent fallback to none
        if confidence >= base_thresholds.get(top_class, 0.25) and max_pattern_score >= 0.2:
            return top_class, confidence, sorted_classes[:2], sorted_probs[:2]
        
        # Fall back to none only if truly uncertain
        return 'none', max(0.40, sorted_probs[sorted_classes == 'none'][0]), sorted_classes[:2], sorted_probs[:2]

def main():
# Create DataFrame
    df = pd.DataFrame(data)

    # Initialize detector
    detector = CognitiveBiasDetector()

# Preprocess the text data
    df['processed_text'] = df['text'].apply(detector.preprocess_text)

# Convert labels to numerical values
    y = detector.label_encoder.fit_transform(df['bias_type'])
    
    # Split with smaller test size
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.15,
        random_state=42,
        stratify=y
    )
    
    # Train the model
    detector.train(df['processed_text'].iloc[train_idx], y[train_idx])

# Evaluate the model
    detector.evaluate(df['processed_text'].iloc[test_idx], y[test_idx])
    
    # Save the model
    detector.save_model('cognitive_bias_detector.joblib')
    
    # Example predictions
    sample_texts = [
        "This new evidence perfectly confirms our existing beliefs about climate change",
        "The original price was $5000, but for you, it's only $3999 today",
        "After hearing about that plane crash, I'm too scared to fly anymore",
        "The research shows mixed results, requiring further investigation",
        "I read a book about quantum physics, now I can explain string theory to anyone",
        "We all saw this economic downturn coming, it was so obvious",
        "The policy changes led to a dramatic increase in crime rates"
    ]
    
    print("\nSample Predictions:")
    for text in sample_texts:
        bias_type, confidence, top_2_classes, top_2_probs = detector.predict_bias(text)
        print(f"\nText: {text}")
        print(f"Top prediction: {bias_type} (confidence: {confidence:.2f})")
        print(f"Second best: {top_2_classes[1]} (confidence: {top_2_probs[1]:.2f})")

if __name__ == "__main__":
    main()
