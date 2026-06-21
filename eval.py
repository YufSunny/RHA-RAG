"""RAG evaluation with ground-truth dataset.

Reads ``test/ground_truth.json``, runs the pipeline for each question, then
computes LLM-as-judge metrics (faithfulness, answer_relevancy, answer_correctness)
and retrieval metrics (context_precision, context_recall via source overlap).

Usage:
    python eval.py      # writes eval_report.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

os.chdir(Path(__file__).parent)

# Load .env if present.
_env = Path(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from langchain_core.messages import HumanMessage, SystemMessage

from rha_rag.llm import models, create_llms
from rha_rag.pipeline import load_all_documents, create_vectorstore
from rha_rag.graph import build_graph
from config import LLM_MODEL


# ── LLM judge ──────────────────────────────────────────────────
_llm, _ = create_llms()
if _llm is None:
    print("FATAL: no LLM available — set OPENAI_API_KEY")
    sys.exit(1)


def _judge(prompt: str) -> str:
    resp = _llm.invoke([SystemMessage(content="Answer concisely. Reply with only the requested format."),
                        HumanMessage(content=prompt)])
    return resp.content.strip()


# ── LLM metrics ────────────────────────────────────────────────

def faithfulness(answer: str, contexts: list[str]) -> float:
    """Are the claims in *answer* supported by the provided *contexts*?"""
    if not answer or not contexts:
        return 0.0
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Score how much of the Answer below is directly supported by the Context. "
        "Reply with a single number from 0.0 (none supported / hallucinated) to "
        "1.0 (every claim supported).\n\n"
        f"Answer:\n{answer}\n\nContext:\n{ctx}"
    )
    try: return float(_judge(prompt))
    except ValueError: return 0.0


def answer_relevancy(question: str, answer: str) -> float:
    """How well does the *answer* address the *question*?"""
    if not answer:
        return 0.0
    prompt = (
        "Score how well the Answer addresses the Question. "
        "Reply with a single number from 0.0 (completely off-topic) to "
        "1.0 (fully relevant, complete).\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    try: return float(_judge(prompt))
    except ValueError: return 0.0


def answer_correctness(question: str, answer: str, reference: str) -> float:
    """How factually correct is *answer* compared to the *reference* answer?"""
    if not answer or not reference:
        return 0.0
    prompt = (
        "Score how factually correct the Candidate answer is, compared to the "
        "Reference answer. Ignore differences in wording or style — only judge "
        "factual content. Reply with a single number from 0.0 (completely wrong "
        "or contradictory) to 1.0 (all facts match).\n\n"
        f"Question: {question}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Candidate:\n{answer}"
    )
    try: return float(_judge(prompt))
    except ValueError: return 0.0


# ── Retrieval metrics ──────────────────────────────────────────

def context_precision(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """Fraction of retrieved-chunk sources that match a ground-truth source."""
    if not retrieved_sources:
        return 0.0
    relevant = [r.lower() for r in relevant_sources]
    hits = sum(1 for s in retrieved_sources if any(r in s.lower() for r in relevant))
    return hits / len(retrieved_sources)


def context_recall(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """Fraction of ground-truth sources that appear in any retrieved chunk source."""
    if not relevant_sources:
        return 1.0
    retrieved_lower = [r.lower() for r in retrieved_sources]
    hits = sum(1 for s in relevant_sources if any(s.lower() in r for r in retrieved_lower))
    return hits / len(relevant_sources)


# ── Load ground truth ─────────────────────────────────────────
gt_file = Path("test/ground_truth.json")
dataset = json.loads(gt_file.read_text(encoding="utf-8"))
print(f"Loaded {len(dataset)} questions from {gt_file}")

# ── Build pipeline ────────────────────────────────────────────
print("Indexing documents in test/ ...")
docs = load_all_documents(
    ["test"],
    progress_callback=lambda msg, cur, tot: print(f"  [{cur}/{tot}] {msg}"),
)
print(f"  {len(docs)} documents loaded")

vectorstore, retriever, chunk_count = create_vectorstore(docs)
print(f"  {chunk_count} chunks indexed")

_, grader_model = create_llms()
graph = build_graph(retriever, _llm, grader_model)
print("Graph compiled OK\n")

# ── Run pipeline ──────────────────────────────────────────────
rows = []

for i, entry in enumerate(dataset, 1):
    q = entry["question"]
    ref = entry.get("reference_answer", "")
    rel_srcs = entry.get("relevant_sources", [])
    print(f"[{i}/{len(dataset)}] {q}")

    # Append grounding hint so the LLM stays faithful to retrieved docs.
    q_grounded = q + " by the uploaded files"

    answer = ""
    for chunk in graph.stream(
        {"question": q_grounded, "history": "", "messages": [HumanMessage(content=q_grounded)]}
    ):
        for node_name, update in chunk.items():
            msg = update["messages"][-1]
            content = getattr(msg, "content", str(msg))
            if node_name in ("generate_answer", "generate_query"):
                answer = content

    ret_docs = retriever.invoke(q)
    contexts = [d.page_content for d in ret_docs]
    retrieved_sources = [d.metadata.get("source", "") for d in ret_docs]

    rows.append({
        "question": q, "answer": answer, "contexts": contexts,
        "retrieved_sources": retrieved_sources,
        "reference": ref, "relevant_sources": rel_srcs,
    })
    print(f"  answer: {len(answer)} chars, retrieved: {len(contexts)} chunks "
          f"from {retrieved_sources}")

# ── Score ─────────────────────────────────────────────────────
print(f"\nEvaluating ({LLM_MODEL}) ...")
score_cols = ["faithfulness", "answer_relevancy", "answer_correctness", "context_recall"]

for r in rows:
    q, a, ctxs, ref, rels, retsrcs = (r["question"], r["answer"], r["contexts"],
                                       r["reference"], r["relevant_sources"],
                                       r["retrieved_sources"])
    r["faithfulness"]       = round(faithfulness(a, ctxs), 3)
    r["answer_relevancy"]   = round(answer_relevancy(q, a), 3)
    r["answer_correctness"] = round(answer_correctness(q, a, ref), 3)
    r["context_recall"]     = round(context_recall(retsrcs, rels), 3)
    print(f"  {q[:55]:55s}  "
          f"faith={r['faithfulness']:.2f}  "
          f"rel={r['answer_relevancy']:.2f}  "
          f"correct={r['answer_correctness']:.2f}  "
          f"cr={r['context_recall']:.2f}")

# ── Averages ──────────────────────────────────────────────────
avgs = {c: round(sum(r[c] for r in rows) / len(rows), 3) for c in score_cols}
print(f"\n{'AVERAGE':>55s}  "
      f"faith={avgs['faithfulness']:.2f}  "
      f"rel={avgs['answer_relevancy']:.2f}  "
      f"correct={avgs['answer_correctness']:.2f}  "
      f"cr={avgs['context_recall']:.2f}")

# ── Report ────────────────────────────────────────────────────
report = {
    "timestamp": datetime.now().isoformat(),
    "judge_model": LLM_MODEL,
    "results": rows,
    "averages": avgs,
}
Path("eval_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nReport → eval_report.json")
