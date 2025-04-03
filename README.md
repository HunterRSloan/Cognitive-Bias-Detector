# Cognitive-Bias-Detector
An advanced machine learning model that identifies cognitive biases in text using natural language processing and ensemble learning techniques.
# Overview
This project implements a sophisticated cognitive bias detection system that can identify seven different types of cognitive biases in text:
Confirmation Bias
Anchoring Bias
Availability Bias
Dunning-Kruger Effect
Framing Bias
Hindsight Bias
Neutral (No bias)
# Key Features
Advanced Pattern Recognition: Utilizes sophisticated regex patterns and linguistic markers to identify bias-specific language patterns
Dual TF-IDF Vectorization: Employs two TF-IDF vectorizers with different n-gram ranges for comprehensive feature extraction
Ensemble Learning: Uses a Random Forest Classifier with optimized hyperparameters for robust bias classification
Dynamic Confidence Thresholds: Implements bias-specific confidence thresholds with pattern-based adjustments
Rich Feature Engineering: Combines TF-IDF features with custom pattern-based features for improved accuracy
Comprehensive Preprocessing: Includes specialized text preprocessing with semantic token replacement and bias marker detection
# Technical Details
Built with Python using scikit-learn, NLTK, and pandas
Implements cross-validation and out-of-bag score evaluation
Features detailed visualization of confusion matrices and classification reports
Includes model persistence functionality for easy deployment
# Performance
Achieves high precision and recall for major bias categories
Implements sophisticated handling of edge cases and ambiguous predictions
Uses stratified k-fold cross-validation for robust performance evaluation
# Requirements
Python 3.6+
scikit-learn
NLTK
pandas
numpy
matplotlib
seaborn
# Usage
from cognitive_bias_detector import CognitiveBiasDetector

# Initialize the detector
detector = CognitiveBiasDetector()

# Make predictions
text = "This new evidence perfectly confirms our existing beliefs"
bias_type, confidence, top_2_classes, top_2_probs = detector.predict_bias(text)

# Applications
Content Analysis
Research Validation
Educational Tools
Writing Assistance
Decision-Making Support
Social Media Analysis
# Future Improvements
Integration with web APIs
Support for additional languages
Real-time bias detection
Enhanced pattern recognition
Expanded bias categories
# License
MIT License
# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
