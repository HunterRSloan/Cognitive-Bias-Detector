
# 🧠 Cognitive Bias Detector — Gradio Demo

Classifies short text for likely cognitive biases (confirmation, anchoring, availability, Dunning–Kruger, framing, hindsight, or none) using:
- TF‑IDF features (1–3‑grams)
- RandomForest classifier
- Regex-based **pattern markers** for explainability

> **Status:** Demo-quality. The bundled dataset is tiny and for illustration only.

---

## ✨ Features
- **Interactive UI** (Gradio) with single text and **batch CSV** tabs
- Shows **class probabilities** and detected **pattern markers** (e.g., `confirmation_marker`)
- **Windows-friendly** setup steps
- Ready to deploy on **Hugging Face Spaces**

---

## 🖥️ Quickstart (Windows)

```cmd
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py app.py
```
The app prints a local URL (e.g., http://127.0.0.1:7860). Click to open.

**Temporary public link:** edit the bottom of `app.py` to use `demo.launch(share=True)`.

---

## 📦 Batch CSV demo
Use the included sample: `batch_test_texts.csv` (one column named `text`).  
In the **Batch (CSV)** tab, upload the file and download `batch_predictions.csv`.

---

## 🚀 Deploy to Hugging Face Spaces
1. Create a Space → **SDK = Gradio**.
2. Upload `app.py` and `requirements.txt` (optional: `README.md`, `batch_test_texts.csv`).
3. Commit. If NLTK downloads stall on first run, click **Restart** once.

---

## 🧩 Project structure
```
.
├─ app.py
├─ requirements.txt
├─ batch_test_texts.csv
├─ README.md
├─ MODEL_CARD.md
├─ CONTRIBUTING.md
├─ CHANGELOG.md
├─ LICENSE
└─ .gitignore
```

---

## 🧪 Notes
- Startup uses `n_estimators=500` for snappy load. Increase to `1500–3000` for accuracy once deployed.
- The app currently trains on startup using the bundled examples. Swap in a larger dataset or load a pre-trained `joblib` for production.

---

## ⚖️ License
[MIT](./LICENSE)

---

## 📄 Model card
See [MODEL_CARD.md](./MODEL_CARD.md) for intended use, limitations, and ethics.
