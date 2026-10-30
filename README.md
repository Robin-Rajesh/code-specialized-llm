# 💻 Code Model Evaluation (Lab 7)

A Streamlit app that compares two code-generation models on Python problems and scores their output against a reference solution with BLEU and a CodeBLEU-style metric.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E)

## ✨ What it does

1. Pick a problem: a recursive factorial, or a palindrome check using slicing.
2. Generate a solution from two models:
   - **Model A:** `gpt2` (general-purpose)
   - **Model B:** `Salesforce/codegen-350M-mono` (code-specialized)
3. Compare each output with the reference solution using:
   - **BLEU** (4-gram, NLTK)
   - **CodeBLEU-like score**: n-gram match combined with simple syntax and data-flow checks
4. See which model scored higher, and how long generation took.

If `transformers`/`torch` aren't installed, or the models fail to download, the app falls back to built-in mock outputs so you can still try the scoring.

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tick **"Use real models"** in the app to download and run the Hugging Face models.
