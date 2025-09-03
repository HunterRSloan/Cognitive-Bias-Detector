
# Model Card — Cognitive Bias Detector (Demo)

**Intended use:** educational demo for detecting likely cognitive-bias patterns in short text.
Do not use for medical, legal, hiring, or other high-stakes decisions.

## Model
- Features: TF–IDF (1–3 grams) + rule-based markers (regex)
- Classifier: RandomForest
- Labels: confirmation_bias, anchoring_bias, availability_bias, dunning_kruger, framing, hindsight_bias, none

## Training data
- Small, hand-written examples included in the repo; meant for demonstration only.

## Limitations
- Small dataset → not generalizable.
- Regex markers may miss paraphrases or trigger on false positives.
- Confidence scores are not calibrated probabilities.

## Ethical considerations
- Avoid using outputs to characterize individuals.
- Prefer opt-in, anonymized text and consider removal upon request.
