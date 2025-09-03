
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from typing import List, Tuple, Dict

# --------- NLTK setup (robust, downloads if missing) ---------
def _ensure_nltk():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)

_ensure_nltk()

# --------- Tiny demo dataset (as provided) ---------
data = {
    'text': [
        # Confirmation Bias
        "This confirms what we've always known about their policies",
        "New study supports existing views on climate change",
        "The evidence clearly backs up our original hypothesis",
        "I only read news sources that align with my beliefs",
        "These statistics prove what I've been saying all along",
        "Finally, research that validates our perspective",
        "I prefer media that reinforces my existing beliefs",
        "The data perfectly matches what we expected to find",

        # Framing
        "Crime rates soar under new administration",
        "Economic growth reaches record highs",
        "The glass is half empty",
        "The glass is half full",
        "90% chance of survival with this treatment",
        "10% risk of death with this treatment",
        "Unemployment drops to historic low of 4%",
        "96% of workers still employed during crisis",

        # Anchoring
        "The initial price was $1000, but now it's only $800",
        "Starting salary is $50,000, which is a great deal",
        "The first offer sets the baseline for negotiations",
        "Original price $200, now reduced to $150",
        "Compare to high-end model at $2000",
        "Market value started at $1M, now only $800K",
        "Previous model cost $2000, this one is just $1500",
        "Compared to last year's budget of $1M, this is reasonable",

        # Availability
        "Recent plane crashes make me afraid to fly",
        "After seeing shark attacks on TV, I'm scared of swimming",
        "Hearing about lottery winners makes me think I could win",
        "News about burglaries makes me double-check my locks",
        "Stories about identity theft make me paranoid online",
        "Recent food poisoning cases make me avoid that restaurant",
        "After watching crime shows, I'm scared to walk alone",
        "Reading about successful startups makes me want to quit my job",

        # Dunning-Kruger
        "I've read one book on psychology, so I'm basically an expert",
        "After watching a few YouTube videos, I can fix any car",
        "I know everything there is to know about this topic",
        "How hard could brain surgery be? I've watched medical shows",
        "I could easily run a restaurant after watching cooking shows",
        "One coding tutorial and I'm ready to build the next Facebook",
        "Just learned Python yesterday, ready to build AI systems",
        "Read an article about quantum physics, now I can explain string theory",

        # Hindsight
        "I knew this would happen all along",
        "The signs were obvious from the beginning",
        "We should have seen the market crash coming",
        "It was clear from the start they would break up",
        "The outcome was predictable from day one",
        "Looking back, the warning signs were everywhere",
        "Anyone could have predicted this result",
        "In retrospect, the pattern was clear",

        # Neutral
        "Studies show both positive and negative effects",
        "The research presents balanced findings",
        "The data suggests multiple possible outcomes",
        "Further investigation is needed to draw conclusions",
        "Evidence supports various interpretations",
        "Results are inconclusive and require more study",
        "More research is needed to confirm these findings",
        "The evidence is currently insufficient to draw conclusions"
    ],
    'bias_type': [
        'confirmation_bias','confirmation_bias','confirmation_bias','confirmation_bias',
        'confirmation_bias','confirmation_bias','confirmation_bias','confirmation_bias',

        'framing','framing','framing','framing','framing','framing','framing','framing',

        'anchoring_bias','anchoring_bias','anchoring_bias','anchoring_bias',
        'anchoring_bias','anchoring_bias','anchoring_bias','anchoring_bias',

        'availability_bias','availability_bias','availability_bias','availability_bias',
        'availability_bias','availability_bias','availability_bias','availability_bias',

        'dunning_kruger','dunning_kruger','dunning_kruger','dunning_kruger',
        'dunning_kruger','dunning_kruger','dunning_kruger','dunning_kruger',

        'hindsight_bias','hindsight_bias','hindsight_bias','hindsight_bias',
        'hindsight_bias','hindsight_bias','hindsight_bias','hindsight_bias',

        'none','none','none','none','none','none','none','none'
    ]
}

# --------- Model ---------
class CognitiveBiasDetector:
    def __init__(self):
        # Vectorizers
        self.vectorizer1 = TfidfVectorizer(
            max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.95,
            analyzer='word', token_pattern=r'\b\w+\b'
        )
        self.vectorizer2 = TfidfVectorizer(
            max_features=1500, ngram_range=(2, 3), min_df=2, max_df=0.95,
            analyzer='word', token_pattern=r'\b\w+\b'
        )

        # Slightly reduced for faster demo startup; increase for accuracy if desired
        self.model = RandomForestClassifier(
            n_estimators=500,  # was 3500 in your script
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight={
                0: 2.5,  # anchoring_bias
                1: 4.0,  # availability_bias
                2: 3.0,  # confirmation_bias
                3: 4.0,  # dunning_kruger
                4: 2.0,  # framing
                5: 3.5,  # hindsight_bias
                6: 2.0   # none
            },
            n_jobs=-1,
            criterion='entropy',
            bootstrap=True,
            max_features='sqrt',
            oob_score=False
        )

        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Keep a list of marker tokens for explanations
        self._markers = [
            'confirmation_marker', 'prior_belief_marker', 'selective_exposure_marker',
            'extreme_change_marker', 'framing_perspective_marker', 'outcome_framing_marker',
            'anchor_point_marker', 'price_reduction_marker', 'anchoring_effect_marker',
            'recency_marker', 'fear_response_marker', 'availability_influence_marker',
            'overconfidence_marker', 'limited_exposure_marker', 'task_simplification_marker',
            'hindsight_marker', 'retrospect_marker', 'prediction_claim_marker'
        ]

    def preprocess_text(self, text: str) -> str:
        text = text.lower()

        # Numbers & currency -> semantic buckets
        text = re.sub(r'\d+%', ' percentage_value ', text)
        text = re.sub(r'\$\d+[km]?', ' price_value ', text)
        text = re.sub(r'\d+k', ' thousand_value ', text)
        text = re.sub(r'\d+m', ' million_value ', text)

        # Confirmation
        text = re.sub(r'(confirms?|supports?|validates?|proves?|back(s|ed) up|align(s|ed)?)', ' confirmation_marker ', text)
        text = re.sub(r'(always|already|consistently) (known|believed|thought|said|claimed)', ' prior_belief_marker ', text)
        text = re.sub(r'(only|exclusively) (read|trust|believe|follow)', ' selective_exposure_marker ', text)

        # Framing
        text = re.sub(r'(soars?|plummets?|crashes?|dramatic(ally)?|surge|plunge)', ' extreme_change_marker ', text)
        text = re.sub(r'(half empty|half full|positive|negative|good|bad) (side|aspect|view|outlook)', ' framing_perspective_marker ', text)
        text = re.sub(r'(chance|risk|probability) of (survival|death|success|failure)', ' outcome_framing_marker ', text)

        # Anchoring
        text = re.sub(r'(initial|original|starting|first|baseline) (price|value|offer|point)', ' anchor_point_marker ', text)
        text = re.sub(r'(now only|reduced to|special price|compared to|reference)', ' price_reduction_marker ', text)
        text = re.sub(r'(sets?|establish(es)?|determine(s)?) (baseline|standard|benchmark)', ' anchoring_effect_marker ', text)

        # Availability
        text = re.sub(r'(after seeing|recent|lately|in the news|media|headlines|hearing about)', ' recency_marker ', text)
        text = re.sub(r'(afraid|scared|worried|paranoid|anxious|concerned|fear) (of|about|that)', ' fear_response_marker ', text)
        text = re.sub(r'(makes? me|causing|leads? to|got me) (think|believe|feel|worry|concerned)', ' availability_influence_marker ', text)

        # Dunning-Kruger
        text = re.sub(r'(basically|easily|just|simply|obviously|already|now) (an expert|understand|know|master|explain|teach|build)', ' overconfidence_marker ', text)
        text = re.sub(r'(one|few|couple|brief|just|recently|yesterday) (book|video|tutorial|course|lesson|article|day)', ' limited_exposure_marker ', text)
        text = re.sub(r'(how hard could|cant be that|easy to|simple enough|ready to|qualified to)', ' task_simplification_marker ', text)

        # Hindsight
        text = re.sub(r'(knew|obvious|clear|predictable|saw) .*(all along|from the (start|beginning)|coming)', ' hindsight_marker ', text)
        text = re.sub(r'(should have (known|seen|predicted)|looking back|in retrospect)', ' retrospect_marker ', text)
        text = re.sub(r'(saw this|expected|anticipated|could have predicted) .*(coming|happening|occurring)', ' prediction_claim_marker ', text)

        # Special tokens
        text = re.sub(r'(always|never|every|all|none|impossible|definitely)', r'absolute_term_\1', text)
        text = re.sub(r'(clearly|obviously|certainly|undoubtedly|absolutely)', r'certainty_term_\1', text)
        text = re.sub(r'(but now|only|just|merely|simply)', r'contrast_term_\1', text)

        tokens = word_tokenize(text)

        important_words = {
            'no','not','none','nothing','never','only','but','however',
            'although','despite','always','every','all','clearly','obviously',
            'certainly','must','should','would','could','might','may','perhaps',
            'possibly','probably','confirms','proves','supports','validates',
            'believe','think','know','understand','expert','easily','obvious',
            'clear','predictable','knew','saw','afraid','scared','worried',
            'recent','news','price','value','offer','deal','special','dramatic',
            'soar','plummet','crash','increase','decrease','rise','fall','more','less'
        }

        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if (t not in self.stop_words or t in important_words) and len(t) > 1
        ]
        return ' '.join(tokens)

    def extract_features(self, texts: List[str]):
        X1 = self.vectorizer1.transform(texts)
        X2 = self.vectorizer2.transform(texts)

        pattern_features = []
        for text in texts:
            feats = {
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
                'has_availability_pattern': 1 if any(m in text for m in ['recency_marker','fear_response_marker','availability_influence_marker']) else 0,
                'has_dunning_kruger_pattern': 1 if any(m in text for m in ['overconfidence_marker','limited_exposure_marker','task_simplification_marker']) else 0,
                'has_framing_pattern': 1 if any(m in text for m in ['extreme_change_marker','framing_perspective_marker','outcome_framing_marker']) else 0,
                'confirmation_with_certainty': 1 if 'confirmation_marker' in text and 'certainty_term' in text else 0,
                'availability_with_fear': 1 if 'recency_marker' in text and 'fear_response_marker' in text else 0,
                'dunning_kruger_with_confidence': 1 if 'limited_exposure_marker' in text and 'overconfidence_marker' in text else 0,
                'framing_with_extreme': 1 if 'framing_perspective_marker' in text and 'extreme_change_marker' in text else 0,
                'has_balanced_perspective': 1 if any(w in text for w in ['both','multiple','various']) else 0,
                'has_uncertainty_marker': 1 if any(w in text for w in ['possibly','perhaps','maybe','might']) else 0
            }
            pattern_features.append(list(feats.values()))
        pattern_features = np.array(pattern_features)
        return np.hstack([X1.toarray(), X2.toarray(), pattern_features])

    def train(self, texts: pd.Series, y: np.ndarray):
        self.vectorizer1.fit(texts)
        self.vectorizer2.fit(texts)
        X = self.extract_features(texts.tolist())
        self.model.fit(X, y)

    def markers_in(self, processed_text: str) -> List[str]:
        return sorted({m for m in self._markers if m in processed_text})

    def predict_bias(self, text: str) -> Tuple[str, float, List[str], Dict[str, float]]:
        processed = self.preprocess_text(text)
        feats = self.extract_features([processed])
        proba = self.model.predict_proba(feats)[0]
        # Correct mapping from proba indices -> string labels
        class_codes = self.model.classes_
        class_names = self.label_encoder.inverse_transform(class_codes)
        prob_map = {name: float(p) for name, p in zip(class_names, proba)}
        # Top prediction
        top_label = max(prob_map, key=prob_map.get)
        top_conf = prob_map[top_label]
        markers = self.markers_in(processed)
        return top_label, float(top_conf), markers, prob_map

# --------- Train on startup (tiny set -> quick) ---------
df = pd.DataFrame(data)
detector = CognitiveBiasDetector()
df['processed_text'] = df['text'].apply(detector.preprocess_text)
y = detector.label_encoder.fit_transform(df['bias_type'])
# Stratify just to be neat; not used later
train_idx, _ = train_test_split(np.arange(len(df)), test_size=0.15, random_state=42, stratify=y)
detector.train(df['processed_text'].iloc[train_idx], y[train_idx])

# --------- Gradio functions ---------
def predict_single(text: str):
    text = (text or "").strip()
    if not text:
        return {"": 0.0}, "—", [], "Please enter text to analyze."
    label, conf, markers, prob_map = detector.predict_bias(text)
    # Build a human-friendly explanation
    if markers:
        exp = f"Detected pattern markers: {', '.join(markers)}"
    else:
        exp = "No explicit pattern markers found; decision based on TF‑IDF features."
    return prob_map, f"{label} ({conf:.2f})", markers, exp

def predict_batch(file_obj):
    if file_obj is None:
        return pd.DataFrame(), None
    df_in = pd.read_csv(file_obj.name)
    # Try common text column names
    col = None
    for c in df_in.columns:
        if str(c).strip().lower() in {"text","sentence","utterance","content","input"}:
            col = c
            break
    if col is None:
        raise gr.Error("CSV must have a 'text' column (or one named sentence/utterance/content/input).")
    rows = []
    for t in df_in[col].astype(str).tolist():
        label, conf, markers, prob_map = detector.predict_bias(t)
        # top 2
        top2 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:2]
        rows.append({
            "text": t,
            "prediction": label,
            "confidence": round(conf, 4),
            "second_best": top2[1][0] if len(top2) > 1 else "",
            "second_confidence": round(top2[1][1], 4) if len(top2) > 1 else 0.0,
            "markers": ", ".join(markers)
        })
    out_df = pd.DataFrame(rows)
    # Save a downloadable copy
    out_path = "batch_predictions.csv"
    out_df.to_csv(out_path, index=False)
    return out_df, out_path

examples = [
    "This new evidence perfectly confirms our existing beliefs about climate change",
    "The original price was $5000, but for you, it's only $3999 today",
    "After hearing about that plane crash, I'm too scared to fly anymore",
    "The research shows mixed results, requiring further investigation",
    "I read a book about quantum physics, now I can explain string theory to anyone",
    "We all saw this economic downturn coming, it was so obvious",
    "The policy changes led to a dramatic increase in crime rates"
]

with gr.Blocks(title="Cognitive Bias Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Cognitive Bias Detector\nEnter text to classify potential cognitive biases.")
    with gr.Tab("Single Text"):
        inp = gr.Textbox(label="Input text", placeholder="Paste a sentence or short paragraph...", lines=5)
        btn = gr.Button("Analyze", variant="primary")
        out_label = gr.Label(label="Predicted bias (scores for all classes)", num_top_classes=7)
        out_top = gr.Textbox(label="Top prediction", interactive=False)
        out_markers = gr.JSON(label="Detected pattern markers")
        out_exp = gr.Markdown()
        btn.click(predict_single, inputs=inp, outputs=[out_label, out_top, out_markers, out_exp])
        gr.Examples(label="Try these", examples=[[e] for e in examples], inputs=inp, examples_per_page=7)

    with gr.Tab("Batch (CSV)"):
        gr.Markdown("Upload a CSV with a **text** column. You'll get a table and a downloadable CSV of predictions.")
        file_in = gr.File(label="Upload CSV")
        btn2 = gr.Button("Analyze CSV")
        table_out = gr.Dataframe(label="Predictions")
        file_out = gr.File(label="Download predictions.csv")
        btn2.click(predict_batch, inputs=file_in, outputs=[table_out, file_out])

    gr.Markdown(
        "Model: RandomForest + TF‑IDF + rule‑based markers • Demo dataset is small (for illustration). "
        "For production, train on a larger, balanced corpus."
    )

if __name__ == "__main__":
    demo.launch()
