# 🧠 Cognitive Bias Detector

A machine learning–powered tool that identifies cognitive biases in text using natural language processing and ensemble classification. Built to support research, education, and content evaluation with insights grounded in psychological theory.

---

## 🔍 Overview

The Cognitive Bias Detector is an AI model trained to detect **seven** cognitive biases from natural language input:

- **Confirmation Bias**
- **Anchoring Bias**
- **Availability Bias**
- **Dunning-Kruger Effect**
- **Framing Bias**
- **Hindsight Bias**
- **Neutral** (no bias detected)

This project combines statistical NLP with custom rule-based pattern recognition to deliver meaningful and interpretable results.

---

## 🚀 Key Features

- **Dual TF-IDF Vectorization**: Combines two n-gram ranges for deeper linguistic analysis
- **Ensemble Learning Model**: Uses a Random Forest classifier with hyperparameter optimization
- **Regex Pattern Matching**: Detects psychologically informed markers and bias indicators
- **Bias-Specific Confidence Thresholds**: Adjusted dynamically for each bias type
- **Rich Feature Engineering**: Includes custom features that enhance prediction clarity
- **Visual Insights**: Confusion matrices and classification reports for model interpretability
- **Reusable API-like Module**: Easily integrated into other tools or pipelines

---

## 🛠️ Tech Stack

- Python 3.6+
- `scikit-learn`, `pandas`, `NLTK`, `numpy`
- `matplotlib`, `seaborn` (for evaluation)
- Model persistence using `joblib`

---

## 📈 Model Performance

- High **precision and recall** across key bias categories
- Robust handling of **ambiguous or borderline cases**
- Evaluated using **stratified k-fold cross-validation**

---

## 💡 Example Usage

```python
from cognitive_bias_detector import CognitiveBiasDetector

detector = CognitiveBiasDetector()

text = "This new evidence perfectly confirms our existing beliefs"
bias_type, confidence, top_2_classes, top_2_probs = detector.predict_bias(text)

print(bias_type, confidence)

```
---

## Applications

- Academic research & validation

- Social media content analysis

- Writing assistance & journalism

- Bias-aware decision making

- Psychological and educational tools

- Social Media Analysis

---

## Future Improvements
- Deploy as a Gradio web app for public use

- Add multilingual support (starting with Spanish + French)

- Expand bias taxonomy to include 12–15 types

- Integrate fine-tuned transformer models (e.g., BERT or RoBERTa)

- Provide rewording or bias-neutral suggestions

---

## License

MIT License

---

## Installation

- pip install -r requirements.txt

---


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
