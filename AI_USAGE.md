# Disclosure of AI Tool Usage

**Candidate:** Jayesh  
**Role:** Platform Engineering Internship, Predixion AI  
**Assignment:** Production-Oriented AI Research Agent  

In accordance with the assignment guidelines and submission instructions, this document provides a comprehensive and transparent disclosure of all AI tools used throughout the development of this project, how they were utilized, and the human verification and engineering decisions applied.

---

## 1. Summary of AI Tools Used

| Tool | Purpose / Area of Use | Nature of Assistance |
|---|---|---|
| **Anthropic Claude & Google Gemini (via AI coding assistant)** | Scaffolding initial file templates, drafting unit test fixtures, generating boilerplate models, documentation drafting | Interactive pair programming, code review, refactoring assistance |
| **Groq / Llama 3 API (OpenAI-compatible SDK)** | Runtime agent inference: small model (`llama-3.1-8b-instant`) for extraction & filtering, large model (`llama-3.3-70b-versatile`) for planning & synthesis | LLM components powering the agent's reasoning loop |

---

## 2. Granular Breakdown: What Was AI-Assisted vs. Human-Designed

### A. Architectural & System Design (Human-Led)
- **Design Philosophy:** The decision to avoid heavyweight frameworks (LangChain, LangGraph, CrewAI) in favor of a hand-rolled, deterministic Python state machine with Pydantic contracts was a deliberate human architectural decision.
- **Security & Threat Model:** Designing the untrusted document boundary, SSRF IP-blocklist filtering, and heuristic injection scanner pattern was human-specified to reflect production platform engineering standards.
- **Hallucination Prevention:** The requirement that synthesis sees *only* verified evidence and that quotes must pass deterministic verbatim matching against source text was conceived to ensure auditability.
- **Mathematical Confidence Model:** The multi-factor scoring formula (weighing domain quality, coverage, quote verification rate, consistency, and penalties) was human-designed.

### B. Code Implementation & Scaffolding (AI-Assisted)
- **Pydantic Data Contracts (`schemas.py`):** AI tools were used to rapidly generate Pydantic v2 schemas from the initial data contract specification.
- **Unit & Integration Test Cases (`tests/`):** AI was used to draft unit tests covering SSRF edge cases (loopback, link-local, AWS metadata endpoints), regex injection patterns, and mock fixtures.
- **Regex Heuristics (`safety.py`):** AI helped assemble standard prompt-injection signature patterns (instruction overrides, role switches, system prompt extractors).
- **Documentation & Reporting:** Initial outlines for `docs/architecture.md`, `EVALUATION.md`, and `README.md` were scaffolded with AI assistance and subsequently edited and verified.

---

## 3. Human Verification, Refactoring, and Engineering Ownership

All AI-generated code was subjected to rigorous human inspection, manual debugging, and automated verification:
1. **Python 3.10 vs 3.11 Compatibility:** Identified and resolved cross-version incompatibilities (e.g. `asyncio.TimeoutError` inheritance differences in Python 3.10 vs 3.11, replacing `datetime.UTC` with `timezone.utc`).
2. **Deterministic Verbatim Quote Matching:** Fixed normalization logic in `evidence.py` to properly handle curly quotes (`\u201c`, `\u201d`) and multi-space variations using explicit Unicode mapping tables and `difflib.SequenceMatcher` sliding window comparisons.
3. **Resilience & Chaos Engineering:** Verified that failure modes (`search_down`, `timeout`, `empty`, `injection`) reliably recover or degrade gracefully without crashing or throwing unhandled exceptions.
4. **Test Suite Execution:** Maintained 100% pass rate across 25 unit and integration tests under `pytest`, and zero violations under `ruff` linting.

Every line of code and architectural decision in this repository is understood, owned, and ready to be explained in depth during technical interviews.

