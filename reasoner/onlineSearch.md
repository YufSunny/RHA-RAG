# SOP-003: Online Literature Search

**Version:** 1.0  
**Effective Date:** 2026-06-10  
**Owner:** Researcher Agent

---

## 1. Purpose

To define the standardized procedure for searching, fetching, and storing online academic literature relevant to a research problem.

## 2. Scope

Applies when the local knowledge base is insufficient to answer the research question, or when the user explicitly requests an online literature search. Complements SOP-001 and SOP-002.

## 3. Prerequisites

- A clarified research problem (from SOP-001, Step 1).
- Configured academic database API endpoint (`ACADEMIC_DATABASE_API_ENTRYPOINT`).
- Target citation style specified (`CITATION_STYLE`).
- Target storage folder for fetched resources (`ONLINE_RESOURCE_FOLDER`).

---

## 4. Procedure

### Step 1 — Query Formulation

**Objective:** Convert the clarified research problem into effective search queries.

1. Extract key concepts and their synonyms from the goal-driven logical statements (SOP-001, Step 1).
2. Construct Boolean search queries combining terms with AND/OR.
3. Prioritize queries by expected recall (broad → narrow).

**Example:**
```
Problem: "What are the effects of X on Y in context Z?"
Query 1 (broad):  X AND Y
Query 2 (narrow): "X" AND "Y" AND Z
Query 3 (specific): effect of X on Y in Z
```

### Step 2 — Database Search

**Objective:** Execute queries against academic databases.

1. Submit each query to the configured `ACADEMIC_DATABASE_API_ENTRYPOINT`.
2. Collect results: title, authors, year, abstract, DOI/URL, and venue.
3. Deduplicate across queries (match on DOI or title+author).
4. Rank results by relevance (title/abstract match to research problem).

### Step 3 — Result Filtering

**Objective:** Select only the most relevant papers.

**Criteria (score each paper 0–1 on each):**
- **Relevance:** Abstract directly addresses the research problem.
- **Recency:** Published within the last 5 years (unless seminal work).
- **Credibility:** Peer-reviewed venue or reputable preprint server.
- **Accessibility:** Full text is retrievable.

**Threshold:** Retain papers scoring ≥ 0.5 average across all criteria.

### Step 4 — Full-Text Fetching

**Objective:** Retrieve and store the full text of selected papers.

1. For each retained paper, attempt to fetch the full text (PDF or HTML).
2. If full text is behind a paywall, store the abstract and metadata with a note.
3. Save each fetched resource to `ONLINE_RESOURCE_FOLDER` with the naming convention:

```
{ONLINE_RESOURCE_FOLDER}/{FirstAuthorLastName}_{Year}_{ShortTitle}.{ext}
```

### Step 5 — Citation Extraction

**Objective:** Record citations in the specified format.

1. For each fetched paper, generate a full citation in `CITATION_STYLE`.
2. Store citations in a structured bibliography file:

```
{ONLINE_RESOURCE_FOLDER}/bibliography.{bib|json|yaml}
```

3. Annotate each entry with its retrieval status: `FULL_TEXT` | `ABSTRACT_ONLY` | `UNAVAILABLE`.

---

## 5. Output

- A folder (`ONLINE_RESOURCE_FOLDER`) containing fetched papers.
- A `bibliography` file with formatted citations.
- A search report summarizing:
  - Queries executed.
  - Papers found, filtered, and fetched.
  - Papers excluded with reasons.

---

## 6. Quality Criteria

- All queries are derived from the clarified problem (traceable).
- Filtering criteria are applied consistently.
- Citations are complete and correctly formatted.
- The search report allows reproducibility (another researcher could re-run the same queries and get the same results).

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-10 | Researcher Agent | Initial SOP |
