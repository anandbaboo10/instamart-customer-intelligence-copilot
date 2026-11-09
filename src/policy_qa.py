"""Run grounded policy question answering over generated policy documents using mock_llm.
Run from the repository root:
    python src/policy_qa.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

POLICY_DIR = Path("data/policy")
OUTPUT_DIR = Path("outputs")
TOP_K = 3

SYSTEM_PROMPT = """POLICY ASSISTANT
Answer only from the supplied policy context. Do not add policies, assumptions, or outside knowledge.
If the answer is not present in the context, say: I could not find that information in the policy documents.
Use a concise answer and preserve the policy's qualification or limit when one applies."""

TEST_CASES = [
    {
        "question": "Which orders are eligible for a refund?",
        "expected_source": "refund_policy.txt",
    },
    {
        "question": "What refund is available when delivery takes more than 45 minutes?",
        "expected_source": "refund_policy.txt",
    },
    {
        "question": "What is the maximum automatic refund amount?",
        "expected_source": "refund_policy.txt",
    },
    {
        "question": "Can a cancelled order receive a refund?",
        "expected_source": "refund_policy.txt",
    },
    {
        "question": "What delivery time does the delivery SLA target?",
        "expected_source": "delivery_sla.txt",
    },
    {
        "question": "When is a delivery considered delayed?",
        "expected_source": "delivery_sla.txt",
    },
    {
        "question": "What customer information must support agents mask?",
        "expected_source": "privacy.txt",
    },
    {
        "question": "Can an agent repeat a customer's full card number in a reply?",
        "expected_source": "privacy.txt",
    },
    {
        "question": "Which refund cases require escalation to a human lead?",
        "expected_source": "escalation.txt",
    },
    {
        "question": "What should happen when a customer reports a fraud or account takeover claim?",
        "expected_source": "escalation.txt",
    },
    {
        "question": "How can an employee change a parking pass?",
        "expected_source": "none",
    },
]


def load_policy_chunks() -> pd.DataFrame:
    if not POLICY_DIR.exists():
        raise FileNotFoundError("Policy documents are missing. Run generate_data.py first.")
    rows = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        # Documents are short, but sentence chunks make retrieval inspectable and reusable.
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        for index, sentence in enumerate(sentences, start=1):
            rows.append({"source_file": path.name, "chunk_id": f"{path.stem}_{index}", "chunk_text": sentence})
    if not rows:
        raise ValueError("No non-empty .txt policy documents were found.")
    return pd.DataFrame(rows)


def retrieve(question: str, chunks: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Rank policy chunks with TF-IDF cosine similarity; retain only positive matches."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([*chunks["chunk_text"].tolist(), question])
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    ranked = chunks.copy()
    ranked["retrieval_score"] = scores
    ranked = ranked[ranked["retrieval_score"] > 0].sort_values("retrieval_score", ascending=False).head(top_k)
    return ranked.reset_index(drop=True)


def answer_question(question: str, retrieved: pd.DataFrame) -> str:
    if retrieved.empty:
        return "I could not find that information in the policy documents."
    context = "\n".join(
        f"[{row.source_file}] {row.chunk_text}" for row in retrieved.itertuples(index=False)
    )
    response = mock_llm.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ]
    )
    return (response.choices[0].message.content or "").strip()


def evaluate(chunks: pd.DataFrame) -> pd.DataFrame:
    mock_llm.reset()
    rows = []
    refusal = "I could not find that information in the policy documents."
    for case in TEST_CASES:
        retrieved = retrieve(case["question"], chunks)
        answer = answer_question(case["question"], retrieved)
        retrieved_sources = "; ".join(retrieved["source_file"].tolist()) if not retrieved.empty else "none"
        expected_source = case["expected_source"]
        source_match = (expected_source == "none" and retrieved.empty) or (expected_source in set(retrieved["source_file"]))
        if retrieved.empty:
            answer_supported = answer == refusal
        else:
            answer_supported = answer in set(retrieved["chunk_text"])
        rows.append(
            {
                "question": case["question"],
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "top_retrieval_score": round(float(retrieved.iloc[0]["retrieval_score"]), 4) if not retrieved.empty else 0.0,
                "answer": answer,
                "source_match": source_match,
                "answer_supported_by_retrieved_context": answer_supported,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_policy_chunks()
    results = evaluate(chunks)

    chunks.to_csv(OUTPUT_DIR / "policy_retrieval_chunks.csv", index=False)
    results.to_csv(OUTPUT_DIR / "policy_qa_results.csv", index=False)

    metrics = {
        "test_cases": int(len(results)),
        "source_retrieval_accuracy": round(float(results["source_match"].mean()), 4),
        "grounded_answer_rate": round(float(results["answer_supported_by_retrieved_context"].mean()), 4),
        "unsupported_answers": int((~results["answer_supported_by_retrieved_context"]).sum()),
        "top_k": TOP_K,
    }
    with (OUTPUT_DIR / "policy_evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    lines = [
        "# Grounded Policy QA Evaluation",
        "",
        "## Retrieval design",
        "- Source documents: generated `.txt` files in `data/policy`.",
        "- Chunking: each policy sentence is a traceable chunk with its source filename.",
        "- Retrieval: TF-IDF cosine similarity; only positive-score chunks are passed to the answerer.",
        f"- Context limit: top {TOP_K} chunks.",
        "- Generation guardrail: the assistant is instructed to use only retrieved context and to refuse when no context is retrieved.",
        "",
        "## Evaluation results",
        f"- Test cases: {metrics['test_cases']}",
        f"- Source retrieval accuracy: {metrics['source_retrieval_accuracy']:.2%}",
        f"- Grounded-answer rate: {metrics['grounded_answer_rate']:.2%}",
        f"- Unsupported answers: {metrics['unsupported_answers']}",
        "",
        "## Test-case traceability",
        "| Question | Expected source | Retrieved source(s) | Source match | Grounded answer |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        question = str(row.question).replace("|", "\\|")
        sources = str(row.retrieved_sources).replace("|", "\\|")
        lines.append(
            f"| {question} | {row.expected_source} | {sources} | {row.source_match} | {row.answer_supported_by_retrieved_context} |"
        )
    (OUTPUT_DIR / "policy_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Policy chunks indexed: {len(chunks)}")
    print(f"Source retrieval accuracy: {metrics['source_retrieval_accuracy']:.3f}")
    print(f"Grounded-answer rate: {metrics['grounded_answer_rate']:.3f}")
    print("Wrote outputs/policy_qa_results.csv and outputs/policy_evaluation.md")


if __name__ == "__main__":
    main()
