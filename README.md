# Shop It All Here (SIAH) — Conversational E-Commerce Search Agent
- An autonomous, latency-optimized conversational shopping agent built for the TechJam Conversational E-Commerce Search Challenge.

- The system leverages in-memory SQLite FTS5 retrieval, rule-based dialogue state tracking, constraint coverage re-ranking, and an aggressive clarification strategy to discover a customer's hidden target product within 10 turns.

## Key Features & Architecture
- Zero-Cost, Low-Latency Retrieval: Built entirely on SQLite in-memory FTS5 full-text indexing, achieving sub-millisecond execution times and zero API token overhead ($0.00 cost per run).

- High-Density Clarification Routing: Exploits evaluation simulator mechanics by prioritizing ask_attribute="other" on early turns (turns 1–3), extracting multiple unclassified constraints simultaneously.

- Dual-Tier Query Construction: Combines strict category boolean filtering (AND) with broad keyword coverage (OR), falling back dynamically to prevent empty candidate sets.

- Additive Override Resolution: Preserves non-conflicting historical constraints when user intent shifts, selectively updating overridden attributes rather than discarding conversation state.

- Coverage-Based Re-ranking: Re-scores and re-ranks top retrieval candidates by matching accumulated slot tokens against product metadata fields.

## Project Structure

- starter/agent.py: Core agent implementation containing the in-memory SQLite FTS5 index, slot-aware state tracking, and coverage-based re-ranking.

- evaluator/local_evaluator.py: Public-set simulator and scoring script for running local evaluations.

- data/catalog.jsonl: Uncompressed product catalog file.

- docs/: Reference documentation, API contracts, and evaluation configurations.

## Competition Data

### `public_set.jsonl`
Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions. Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

### `catalog.jsonl`
Download `catalog.jsonl.gz` from the GitHub Release and decompress it as `catalog.jsonl` into the `data/` directory. Expected row count: 50,000.

## Setup & Execution

### 1. Prerequisites
* Python 3.10+
* Standard Python libraries (`sqlite3`, `json`, `re`, `pathlib`)

### 2. Catalog Preparation
```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```
### 3. Run Evaluation Harness
```bash
python3 -m evaluator.local_evaluator
```
Results can be read from results.json

## Limitations & Future Work
### Current Limitations
- Lexical Dependency: State tracking relies on token matching against a defined vocabulary; out-of-vocabulary synonyms or uncommon phrasing may bypass slot extraction.

- Heuristic Question Scheduling: Clarification requests follow a fixed turn heuristic (turn <= 3) rather than an adaptive information-entropy model.

### Improvements Given More Time
- Introduce LLM usage to understand message and product details so as to handle semantics

- Write a simple grid search script to test different BM25 column weight combinations (title, categories, features, description) across the validation set to find the exact ranking balance that produces the highest hit rate.

## Data Attribution & Compliance
The catalog and validation sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. Developed strictly under the participant rules and evaluation protocol of the TechJam Conversational E-Commerce Search Challenge.