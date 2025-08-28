import streamlit as st
import nltk
from nltk.translate.bleu_score import sentence_bleu
import time

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt', quiet=True)

PROBLEMS = {
    "Calculate Factorial (Recursive)": {
        "prompt": "def factorial(n):\n    \"\"\"Return the factorial of n using recursion.\"\"\"\n",
        "reference": """def factorial(n):
    \"\"\"Return the factorial of n using recursion.\"\"\"
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)"""
    },
    "Check for Palindrome (Slicing)": {
        "prompt": "def is_palindrome(s):\n    \"\"\"Check if a string is a palindrome using slicing.\"\"\"\n",
        "reference": """def is_palindrome(s):
    \"\"\"Check if a string is a palindrome using slicing.\"\"\"
    s = ''.join(filter(str.isalnum, s)).lower()
    return s == s[::-1]"""
    }
}

def safe_load_pipelines():
    try:
        gen1 = pipeline('text-generation', model='gpt2', max_new_tokens=128)
        gen2 = pipeline('text-generation', model='Salesforce/codegen-350M-mono', max_new_tokens=128)
        return gen1, gen2
    except Exception:
        return None, None

def mock_generator(prompt, variant=1):
    if "factorial" in prompt:
        if variant == 1:
            return "def factorial(n):\n    if n<=1:\n        return 1\n    return n*factorial(n-1)"
        else:
            return "def factorial(n):\n    return 1 if n<=1 else n*factorial(n-1)"
    if "is_palindrome" in prompt:
        if variant == 1:
            return "def is_palindrome(s):\n    s=''.join(filter(str.isalnum,s)).lower()\n    return s==s[::-1]"
        else:
            return "def is_palindrome(s):\n    s=''.join(ch for ch in s if ch.isalnum()).lower()\n    return s==s[::-1]"
    return prompt + "\n# generated code"

def calculate_bleu(reference: str, candidate: str):
    ref_tokens = [reference.split()]
    cand_tokens = candidate.split()
    try:
        score = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25,0.25,0.25,0.25))
    except Exception:
        score = 0.0
    return float(score)

def calculate_codebleu_like(reference: str, candidate: str):
    bleu = calculate_bleu(reference, candidate)
    syntax = 1.0 if "return" in candidate and "def " in candidate else 0.5
    dataflow = 1.0 if ("if" in candidate and "return" in candidate) else 0.6
    weighted = bleu * 1.1 if "def " in candidate else bleu
    total = (weighted + syntax + dataflow) / 3
    return float(min(total, 1.0)), {"ngram": bleu, "syntax": syntax, "dataflow": dataflow}

st.set_page_config(page_title="Lab 7: Code Model Evaluation", layout="wide")
st.title("Lab 7: Code Model Evaluation")
st.markdown("Select a problem, generate code from two models (or mocks), and compare using BLEU and a CodeBLEU-like score.")

col1, col2 = st.columns([2,1])
with col1:
    selected = st.selectbox("Problem", list(PROBLEMS.keys()))
    use_real = st.checkbox("Use real models (transformers required)", value=False)
    if use_real and not TRANSFORMERS_AVAILABLE:
        st.warning("transformers or torch not available — using mock generation instead.")
        use_real = False

    if st.button("Generate and Evaluate"):
        prompt = PROBLEMS[selected]["prompt"]
        reference = PROBLEMS[selected]["reference"]
        start = time.time()
        if use_real:
            with st.spinner("Loading models and generating..."):
                gen1, gen2 = safe_load_pipelines()
                if gen1 is None or gen2 is None:
                    st.warning("Model load failed — falling back to mock generation.")
                    out1 = mock_generator(prompt, variant=1)
                    out2 = mock_generator(prompt, variant=2)
                else:
                    out1 = gen1(prompt, num_return_sequences=1)[0]['generated_text']
                    out2 = gen2(prompt, num_return_sequences=1)[0]['generated_text']
        else:
            out1 = mock_generator(prompt, variant=1)
            out2 = mock_generator(prompt, variant=2)
        elapsed = time.time() - start

        bleu1 = calculate_bleu(reference, out1)
        bleu2 = calculate_bleu(reference, out2)
        cb1, breakdown1 = calculate_codebleu_like(reference, out1)
        cb2, breakdown2 = calculate_codebleu_like(reference, out2)

        st.markdown("### Outputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Model A")
            st.code(out1, language='python')
            st.write(f"BLEU: {bleu1:.4f}")
            st.write(f"CodeBLEU-like: {cb1:.4f}")
            st.write(f"Breakdown: ngram:{breakdown1['ngram']:.3f}, syntax:{breakdown1['syntax']:.2f}, dataflow:{breakdown1['dataflow']:.2f}")
        with c2:
            st.subheader("Model B")
            st.code(out2, language='python')
            st.write(f"BLEU: {bleu2:.4f}")
            st.write(f"CodeBLEU-like: {cb2:.4f}")
            st.write(f"Breakdown: ngram:{breakdown2['ngram']:.3f}, syntax:{breakdown2['syntax']:.2f}, dataflow:{breakdown2['dataflow']:.2f}")
        with c3:
            st.subheader("Reference")
            st.code(reference, language='python')

        st.markdown("---")
        st.subheader("Summary")
        if cb1 > cb2:
            st.write("Model A performs better by CodeBLEU-like score.")
        elif cb2 > cb1:
            st.write("Model B performs better by CodeBLEU-like score.")
        else:
            st.write("Both models show similar CodeBLEU-like performance.")
        st.write(f"Execution time: {elapsed:.2f} seconds")

with col2:
    st.subheader("Settings & Notes")
    st.write("If transformers/torch are not installed or model download fails, the app uses fast mock outputs.\n")
    st.write("To use real models, enable 'Use real models' and ensure your environment has transformers and torch installed.")
    st.write("Recommended workflow in VS Code: create venv, activate it, install requirements, then run: streamlit run lab7_app.py")