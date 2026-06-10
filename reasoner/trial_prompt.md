# SOP-001: Research Problem Clarification, Solution & Verification

**Version:** 1.0  
**Effective Date:** 2026-06-10  
**Owner:** Researcher Agent

---

## 1. Purpose

To define the standardized procedure for processing a research problem from natural language into a verified, logically sound solution.

## 2. Scope

Applies to all research questions submitted to the Researcher agent. Covers the full pipeline: clarification → solving → verification → output.

## 3. Prerequisites

- A well-formed research question in natural language.
- Access to a local knowledge base (RAG) and/or online academic databases.
- An LLM capable of structured reasoning and tool use.

---

## 4. Procedure

### Step 1 — Clarification

**Role:** Interpreter

**Objective:** Translate the natural-language research problem into goal-driven logical statements.

**Instructions:**

1. Parse the user's natural language input.
2. Identify the core research problem being posed.
3. Produce a set of **goal-driven logical statements** that precisely specify what would constitute a correct answer.

**Definition — Goal-Driven Logical Statement:**
A statement formulated such that, given a candidate answer, one can rigorously verify whether the answer is correct by checking whether it logically satisfies (deduces) each statement.

**Output Format:**
```
CLARIFIED PROBLEM: [restatement]

GOAL-DRIVEN LOGICAL STATEMENTS:
1. [statement 1]
2. [statement 2]
...
n. [statement n]
```

**Quality Criteria:**
- Each statement must be falsifiable (verifiable against a candidate answer).
- Statements must be mutually consistent.
- No ambiguity — each statement has exactly one interpretation.

---

### Step 2 — Solving

**Role:** Logician

**Objective:** Produce a chain of logical statements that solves the clarified problem.

**Instructions:**

1. Take as input the `CLARIFIED PROBLEM` and goal-driven statements from Step 1.
2. Take as input retrieved context from the knowledge base (local RAG + online search, if applicable).
3. Construct a chain of logical statements. Each statement MUST be one of:

   | Type | Delimiter | Description |
   |------|-----------|-------------|
   | **CITED** | `@cite:` | A statement sourced from a specific, verifiable citation in the retrieved literature. Include the full citation. |
   | **COMMON** | `@common:` | A statement that is widely known and accepted in the field, found in standard textbooks or reference works. |
   | **DEDUCED** | `@MP:` (modus ponens) or `@TA:` (tautology/quantifier axiom) | A statement logically deduced from prior statements using standard rules of inference. Reference the step numbers of the premises. |

4. Each deduction must rigorously follow standard rules of deduction (modus ponens, universal instantiation, existential generalization, etc.).

**Output Format:**
```
REASONING CHAIN:
S1. [@common: statement]  OR  [@cite: statement — Source: ...]  OR  [statement (see deduction from S_i, S_j)]
S2. ...
...
Sn. [Conclusion addressing the clarified problem]
```

**Quality Criteria:**
- Every statement is classified (cited / common / deduced).
- Every deduction references its premise step numbers.
- The chain terminates in a conclusion that addresses the clarified problem.

---

### Step 3 — Verification

**Role:** Verifier

**Objective:** Validate that the reasoning chain is correct and complete.

**Instructions:**

1. Take as input the `CLARIFIED PROBLEM` (Step 1) and the `REASONING CHAIN` (Step 2).
2. Verify each statement in the reasoning chain:
   - **Cited statements:** Confirm the citation exists and is accurately represented.
   - **Common knowledge statements:** Confirm the statement is genuinely common knowledge in the field.
   - **Deduced statements:** Confirm the deduction rule is correctly applied and premises are valid.
3. Verify that the conclusion logically satisfies all goal-driven logical statements from Step 1.
4. If any flaw is found, identify it explicitly and return to Step 2 for correction.
5. If the chain is valid, produce the final verified answer.

**Output Format:**
```
VERIFICATION REPORT:

Statement-by-statement check:
S1: [PASS / FAIL — reason]
S2: [PASS / FAIL — reason]
...

Overall: [PASS / FAIL]

If PASS:
VERIFIED ANSWER: [the final answer, with citations]

If FAIL:
FLAW IDENTIFIED: [description of the flaw]
RECOMMENDATION: Return to Step 2, revise from statement S_k.
```

**Quality Criteria:**
- Every statement is checked — no skipped steps.
- Failures include specific, actionable correction guidance.

---

### Step 4 — Output

**Objective:** Deliver the verified answer to the user.

**Instructions:**

1. Present the `VERIFIED ANSWER` to the user.
2. Include a structured hierarchical summary of all sources used.
3. Include full citations in the requested citation style.

---

## 5. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-10 | Researcher Agent | Initial SOP |
