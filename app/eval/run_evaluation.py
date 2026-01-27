# run_evaluation.py
"""Evaluation script for model test scenarios.

This script loads test data for each core AI function, calls the corresponding
API endpoint, and computes evaluation metrics such as BLEU, ROUGE‑L, accuracy,
response time (TTFT) and cost estimation.

The script is intentionally lightweight – it uses the standard library and
optional third‑party packages (`nltk`, `rouge_score`). Install them with:

    pip install nltk rouge-score

The script writes a summary markdown file `evaluation_results.md` in the same
directory.
"""

import argparse
import csv
import json
import time
from pathlib import Path

# Optional imports – if unavailable, metrics will be skipped.
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    nltk = None
    sentence_bleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_csv(path: Path):
    """Load a CSV file returning a list of rows (as dicts)."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def call_api(endpoint: str, payload: dict):
    """Simple wrapper around a POST request.
    In a real environment replace the URL with the actual service URL.
    """
    import requests

    url = f"http://localhost:8000{endpoint}"
    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    return response.json(), elapsed


def compute_bleu(reference: str, hypothesis: str) -> float:
    if not sentence_bleu:
        return 0.0
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return sentence_bleu([ref_tokens], hyp_tokens)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    if not rouge_scorer:
        return {"rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {"rougeL": scores["rougeL"].fmeasure}


# ---------------------------------------------------------------------------
# Evaluation per feature
# ---------------------------------------------------------------------------


def eval_chat(test_cases: list):
    results = []
    for case in test_cases:
        payload = {
            "messages": [{"role": "user", "content": case["prompt"]}],
            "model": "gemini_flash",
        }
        resp, ttft = call_api("/ai/chat", payload)
        answer = resp.get("answer", "")
        bleu = compute_bleu(case["reference"], answer)
        rouge = compute_rouge(case["reference"], answer)
        results.append(
            {
                "id": case["id"],
                "ttft": ttft,
                "bleu": bleu,
                "rougeL": rouge["rougeL"],
                "answer": answer,
            }
        )
    return results


def eval_interview_qa(test_set: list):
    # Similar to chat but with mode=interview_question
    results = []
    for qa in test_set:
        payload = {
            "messages": [{"role": "user", "content": qa["question"]}],
            "model": "gemini_pro",
            "mode": "interview_question",
        }
        resp, ttft = call_api("/ai/chat", payload)
        answer = resp.get("answer", "")
        # Simple exact‑match accuracy for demonstration
        correct = int(answer.strip() == qa["expected_answer"].strip())
        results.append({"id": qa["id"], "ttft": ttft, "accuracy": correct, "answer": answer})
    return results


# Additional evaluation functions (job posting, resume, report) would follow the same pattern.

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation for core AI functions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../test_data"),
        help="Directory containing test data files.",
    )
    args = parser.parse_args()

    # Load test data – paths are examples; adjust to your repo layout.
    chat_cases = load_csv(args.data_dir / "chat_test_cases.csv")
    interview_cases = load_json(args.data_dir / "interview_qa.json")

    chat_res = eval_chat(chat_cases)
    interview_res = eval_interview_qa(interview_cases)

    # Write a simple markdown summary
    out_path = Path(__file__).with_name("evaluation_results.md")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Chat (BLEU / ROUGE‑L)\n")
        for r in chat_res:
            f.write(
                f"- ID {r['id']}: TTFT {r['ttft']:.2f}s, BLEU {r['bleu']:.3f}, ROUGE‑L {r['rougeL']:.3f}\n"
            )
        f.write("\n## Interview Q&A (Accuracy)\n")
        for r in interview_res:
            f.write(f"- ID {r['id']}: TTFT {r['ttft']:.2f}s, Correct {r['accuracy']}\n")
    print(f"Evaluation summary written to {out_path}")


if __name__ == "__main__":
    main()
