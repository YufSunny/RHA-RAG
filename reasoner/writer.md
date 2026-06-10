# SOP-002: Research Paper Writing

**Version:** 1.0  
**Effective Date:** 2026-06-10  
**Owner:** Researcher Agent

---

## 1. Purpose

To define the standardized procedure for producing a research paper from retrieved and verified research findings.

## 2. Scope

Applies when the user requests a formal research paper as output. Covers literature search, structured summarization, and paper drafting.

## 3. Prerequisites

- Completed SOP-001 (clarification → solving → verification) yielding a verified answer with citations.
- Specified research field, problem statement, and target paper style/venue.
- Access to local knowledge base (RAG) and online academic databases.

---

## 4. Procedure

### Step 1 — Literature Search

**Objective:** Gather all relevant sources.

1. **Local Search:** Execute agentic RAG search on the local knowledge base at the configured `LOCAL_KNOWLEDGE_BASE_PATH`.
2. **Online Search:** If the local knowledge base is insufficient, query academic databases via the configured `ACADEMIC_DATABASE_API_ENTRYPOINT`. See SOP-003 for online search procedure.
3. **Deduplication:** Merge local and online results; remove duplicates by title/DOI.

### Step 2 — Structured Summary

**Objective:** Organize retrieved literature into a hierarchical summary with explicit citations.

**Instructions:**

1. Group sources by theme, methodology, or finding.
2. For each group, write a concise synthesis paragraph.
3. Tag every factual claim with its source citation.
4. Identify gaps: note what the literature does not yet address.

**Output Format:**
```
## Structured Literature Summary

### Theme 1: [Name]
- Finding A [Source: Author (Year)]
- Finding B [Source: Author (Year)]
Synthesis: [...]

### Theme 2: [Name]
...

### Gaps Identified
- [gap 1]
- [gap 2]
```

### Step 3 — Paper Drafting

**Objective:** Write the research paper.

**Sections (standard IMRaD):**

1. **Introduction** — State the problem, its significance, and the paper's contribution.
2. **Literature Review** — Present the structured summary (Step 2) in narrative form.
3. **Methodology** — Describe the reasoning approach (per SOP-001).
4. **Results** — Present the verified answer and reasoning chain.
5. **Discussion** — Interpret results, acknowledge limitations, suggest future work.
6. **Conclusion** — Summarize findings and their implications.
7. **References** — Full bibliography in the specified `PAPER_STYLE` format.

**Style:** Adhere to the target `PAPER_STYLE` (e.g., APA 7, IEEE, ACM).

---

## 5. Output

A complete research paper draft with:
- All sections enumerated above.
- Inline citations matching the reference list.
- A hierarchical, citation-traceable argument.

## 6. Quality Criteria

- Every factual claim is backed by a citation.
- The reasoning chain from SOP-001 is faithfully represented.
- The paper conforms to the specified style guide.
- All references are complete and retrievable.

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-10 | Researcher Agent | Initial SOP |
