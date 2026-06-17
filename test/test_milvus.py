"""Test the Milvus vector-store refactor.

Indexes the documents in test/, runs retrieval + the full RHA-RAG graph for the
question in questions.txt, and asserts:
  1. chunking/indexing produced chunks,
  2. retrieval surfaces topology.md for the topology question,
  3. the full graph reaches generate_answer with non-empty content.

Run via:
    python test/test_milvus.py
"""

import os
import sys
from pathlib import Path

# Run from repo root so milvus.db / relative paths resolve.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from rha_rag.llm import models, create_llms
from rha_rag.pipeline import load_all_documents, create_vectorstore
from rha_rag.graph import build_graph
from langchain_core.messages import HumanMessage

TEST_DIR = ROOT / "test"
QUESTIONS_FILE = TEST_DIR / "questions.txt"


def main():
    # --- keys ---
    missing = [m.api_key for _, m in models.items() if not os.environ.get(m.api_key)]
    if missing:
        print(f"FAIL: missing API keys: {missing}")
        sys.exit(1)

    question = QUESTIONS_FILE.read_text(encoding="utf-8").strip()
    print(f"Question: {question}")

    # --- load test docs ---
    print("Loading test documents...")
    docs = load_all_documents(
        [str(TEST_DIR)],
        progress_callback=lambda msg, cur, tot: print(f"  [{cur}/{tot}] {msg}"),
    )
    assert docs, "No documents loaded from test/"
    print(f"  {len(docs)} documents loaded")
    for d in docs:
        print(f"    - {d.metadata.get('name')}")

    # --- index into Milvus ---
    print("Indexing into Milvus...")
    vectorstore, retriever, chunk_count = create_vectorstore(docs)
    assert chunk_count > 0, "No chunks indexed"
    print(f"  {chunk_count} chunks indexed")

    # --- retrieval ---
    print(f"\nRetrieving (k=5) for: {question!r}")
    results = retriever.invoke(question)
    sources = [d.metadata.get("source", "unknown") for d in results]
    print(f"  retrieved {len(results)} chunks from: {sources}")

    assert any("topology" in s.lower() for s in sources), (
        f"Expected topology.md in retrieved sources, got: {sources}"
    )
    print("  PASS: topology.md retrieved")

    # --- full graph ---
    print("\nBuilding graph...")
    response_model, grader_model = create_llms()
    graph = build_graph(retriever, response_model, grader_model)

    print("Streaming graph...")
    final_answer = ""
    nodes_seen = []
    for chunk in graph.stream(
        {"question": question, "history": "", "messages": [HumanMessage(content=question)]}
    ):
        for node_name, update in chunk.items():
            nodes_seen.append(node_name)
            msg = update["messages"][-1]
            content = getattr(msg, "content", str(msg))
            print(f"  -- [{node_name}] {len(content)} chars")
            if node_name == "generate_answer":
                final_answer = content

    print(f"\nNodes seen: {nodes_seen}")
    assert "generate_answer" in nodes_seen, "graph did not reach generate_answer"
    assert final_answer.strip(), "generate_answer produced empty content"
    print("\nPASS: graph reached generate_answer with non-empty content")
    print("\n--- FINAL ANSWER ---")
    print(final_answer)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
