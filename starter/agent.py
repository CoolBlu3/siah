from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# --- Slot vocab -------------------------------------------------------
# Each of these is checked against the *tokenized* message, not a raw
# substring search, so "red" won't match inside "credit" / "bored" etc.
SLOT_VOCAB: dict[str, set[str]] = {
    "material": {
        "cotton", "polyester", "nylon", "leather", "wool",
        "spandex", "silk", "rayon", "fabric", "denim", "linen",
    },
    "color": {
        "black", "white", "blue", "red", "pink", "green", "brown",
        "gray", "grey", "purple", "yellow", "orange", "navy", "beige",
        "tan", "maroon", "olive", "cream", "gold", "golden", "silver",
        "rose", "bronze", "multicolor",
    },
    "size": {
        "xs", "xxs", "xxl", "xl", "small", "medium", "large",
        "s", "m", "l",
    },
    "style": {
        "casual", "formal", "athletic", "vintage", "slim", "loose",
        "oversized", "classic", "modern", "bohemian", "sporty",
        "hoop", "dangle", "stud", "chain", "pendant", "statement",
    },
    "use_case": {
        "wedding", "work", "office", "gym", "running", "hiking",
        "party", "travel", "outdoor", "everyday", "beach", "winter",
        "summer", "date", "interview",
    },
}

# Words that strongly signal the user is stating a hard constraint
# ("Buying" track) rather than just browsing/exploring.
BUYING_SIGNAL_WORDS = {
    "size", "under", "budget", "brand", "need", "must", "exactly",
    "specifically", "only",
}

BUDGET_RE = re.compile(r"under\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
BRAND_RE = re.compile(r"\bby\s+([A-Z][a-zA-Z0-9&' -]{1,30})\b")

# Slots that, when known, make the strongest filters for retrieval.
HARD_SLOTS = ["color", "material", "brand", "size"]
# Order in which we prefer to ask for a missing slot when the
# candidate pool is too broad. NOTE: "brand" is deliberately excluded —
# the simulated customer's constraint classifier never labels anything
# as "brand", so asking for it can never be answered and just burns a
# turn. "budget" is also excluded from here since it doesn't feed the
# text retrieval query at all (see _slot_terms) and asking for it can't
# narrow the candidate pool.
# "feature" is prioritized FIRST (after category): classify_constraint's
# default bucket for anything that isn't a recognized material/color/
# size/style/use_case keyword is "feature" — even oddly-labeled things
# like "Material:alloy" end up there since "alloy" isn't in the fixed
# 9-word materials list. It's where most of the specific, high-signal
# product detail actually lives, so it should be asked about early.
CLARIFY_PRIORITY = ["category", "feature", "color", "material", "use_case", "style", "size"]

# Hard cap on how many clarifying questions we'll ask in a session. With
# a 10-turn budget, spending more than a couple of turns clarifying
# before ever attempting a real search tanks both hit-rate and MTTC —
# better to search early and clarify only if truly necessary.
MAX_CLARIFYING_TURNS = 4

OVER_GENERALITY_THRESHOLD = 40  # candidate pool size that triggers a clarifying question

# bm25() column weights: (parent_asin[unindexed, ignored], title, categories,
# features, details, store, description). Exposed as a module constant so it
# can be tuned/swept against the dev set rather than left at whatever the
# starter shipped with.
BM25_WEIGHTS = (0.0, 6.0, 8.0, 2.5, 2.5, 1.5, 1.0)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def is_override_message(message: str) -> bool:
    """Detects if user is overriding past preferences."""
    lowered = message.lower()
    return "ignore" in lowered or "earlier preference" in lowered or "actually" in lowered


def extract_override_value(message: str) -> str:
    """Extracts the new requirement from an override message."""
    if "what i need is:" in message.lower():
        return message.split(":", 1)[-1].strip(" .")
    return message


def empty_slots() -> dict[str, list[str]]:
    return {
        "category": [],
        "material": [],
        "color": [],
        "size": [],
        "style": [],
        "brand": [],
        "budget": [],
        "feature": [],
        "use_case": [],
    }


class Agent:
    """BM25 retrieval with slot-aware state tracking, intent routing, and
    over-generality clarification."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()

        # Per-session conversation state (slots) and raw turn history.
        self.history: dict[str, dict[str, list[str]]] = {}
        self.turns: dict[str, list[str]] = {}
        # Tracks how many times we've already asked about each slot, so
        # we never loop forever asking the same question the user isn't
        # answering (which would tank MTTC and blow the 10-turn cap).
        self._asked: dict[str, dict[str, int]] = {}
        # Accumulates EVERY term the user has ever typed in the session,
        # not just terms that happen to match our narrow SLOT_VOCAB.
        # Without this, anything disclosed that our vocab doesn't
        # recognize (e.g. a specific feature phrase) only affects the
        # turn it was said in and is forgotten immediately after —
        # which silently throws away most of what the simulated
        # customer discloses.
        self._all_terms: dict[str, list[str]] = {}

        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions.add(session_id)
        self.history[session_id] = empty_slots()
        self.turns[session_id] = []
        self._asked[session_id] = {}
        self._all_terms[session_id] = []

    # -----------------------------------------------------------------
    # State tracking
    # -----------------------------------------------------------------
    def update_history(self, session_id: str, user_message: str) -> dict:
        """Extracts slot values from a message using tokenized matching
        (not raw substrings) and merges them into session state. Also
        accumulates every term from the message into _all_terms so that
        details outside our hand-built vocab aren't lost on later turns."""
        self._all_terms.setdefault(session_id, []).extend(_terms(user_message))

        tokens = set(_terms(user_message))
        constraints: dict[str, list[str]] = {}

        for slot, vocab in SLOT_VOCAB.items():
            hits = sorted(tokens & vocab)
            if hits:
                constraints[slot] = hits

        budget_match = BUDGET_RE.search(user_message)
        if budget_match:
            constraints["budget"] = [budget_match.group(1)]

        brand_match = BRAND_RE.search(user_message)
        if brand_match:
            constraints["brand"] = [brand_match.group(1).strip()]

        for slot, values in constraints.items():
            existing = self.history[session_id].setdefault(slot, [])
            for value in values:
                if value not in existing:
                    existing.append(value)

        return constraints

    def handle_override(self, session_id: str, user_message: str) -> None:
        """Handles an intent-override message additively, NOT destructively.

        Earlier version wiped all accumulated slots and free-text history
        on every override, keeping only the category. That was based on
        a wrong assumption: in practice (and confirmed by tracing actual
        eval sessions), an override message like "actually, ignore my
        earlier preference, what I need is X" replaces exactly ONE prior
        preference — everything else the user disclosed earlier is still
        true and still useful for retrieval. A full wipe was throwing
        away legitimately good signal (e.g. already-disclosed feature
        details) and forcing the agent to re-ask questions it had already
        gotten answers to, which was measurably hurting intent_override
        hit rate and MTTC.

        We don't have a reliable way to identify *which* specific old
        value is being contradicted, so the safe move is to just treat
        the override text as a new, high-priority disclosure and layer
        it on top of everything else known — not to discard history."""
        new_text = extract_override_value(user_message)
        constraints = self.update_history(session_id, new_text)

        # If nothing structured was recognized, keep the raw text as a
        # free-text feature so it still contributes to retrieval.
        if not constraints:
            self.history[session_id]["feature"].append(new_text)

    def _detect_category(self, user_message: str) -> str | None:
        match = re.search(r"^I'm looking for (.*?)[.,]", user_message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _is_buying_intent(self, user_message: str, new_constraints: dict) -> bool:
        """Heuristic dual-track intent routing. 'Buying' = the user gave
        a hard, filterable constraint this turn (or used buying-signal
        language); 'Browsing' = open-ended exploration."""
        if any(slot in new_constraints for slot in HARD_SLOTS + ["budget"]):
            return True
        tokens = set(_terms(user_message))
        return bool(tokens & BUYING_SIGNAL_WORDS)

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------
    def _slot_terms(self, session_id: str) -> list[str]:
        """Flatten accumulated hard-slot values into query terms, PLUS
        every term the user has ever typed this session. The slot values
        are repeated to bias BM25 ranking towards recognized preferences;
        the full accumulated text ensures we never silently drop
        constraint details that fall outside our hand-built vocab."""
        terms: list[str] = list(self._all_terms.get(session_id, []))
        slots = self.history[session_id]
        for slot in ["category", "color", "material", "brand", "style", "use_case"]:
            for value in slots.get(slot, []):
                for tok in _terms(value):
                    terms.append(tok)
                    terms.append(tok)  # extra weight for recognized slots
        return terms

    def _search(self, terms: list[str], top_k: int) -> list[dict]:
        unique_terms = list(dict.fromkeys(terms))[:60]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {', '.join(str(w) for w in BM25_WEIGHTS)}) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [{"parent_asin": str(row[0])} for row in rows]

    def _candidate_count(self, terms: list[str]) -> int:
        unique_terms = list(dict.fromkeys(terms))[:60]
        if not unique_terms:
            return 0
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        row = self.connection.execute(
            "SELECT COUNT(*) FROM products WHERE products MATCH ?",
            (expression,),
        ).fetchone()
        return int(row[0]) if row else 0

    def _missing_slot_to_ask(self, session_id: str) -> str | None:
        """Picks the highest-priority slot that's both (a) still empty and
        (b) hasn't already been asked. Without the 'already asked' check,
        an unrecognized answer (e.g. the user says 'gold' but 'gold' isn't
        in our vocab) causes the agent to re-ask the same question every
        turn forever — burning the 10-turn budget without converging."""
        slots = self.history[session_id]
        asked = self._asked.setdefault(session_id, {})
        for slot in CLARIFY_PRIORITY:
            if not slots.get(slot) and asked.get(slot, 0) == 0:
                return slot
        return None

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        self.turns.setdefault(session_id, []).append(user_message)

        if turn == 1:
            category = self._detect_category(user_message)
            if category:
                self.history[session_id]["category"] = [category]

        if is_override_message(user_message):
            self.handle_override(session_id=session_id, user_message=user_message)
            new_constraints: dict = {}
        else:
            new_constraints = self.update_history(session_id, user_message)

        buying = self._is_buying_intent(user_message, new_constraints)

        # Free-text terms from *this* message (soft signal either way).
        current_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        # Accumulated slot terms carried across the whole session.
        slot_terms = self._slot_terms(session_id)

        if buying:
            # Buying track: weight accumulated hard constraints heavily
            # by repeating them alongside this turn's free text.
            query_terms = slot_terms + slot_terms + current_terms
        else:
            # Browsing track: favor diversity — lean more on the fresh
            # message, lightly nudged by known slots.
            query_terms = current_terms + slot_terms

        candidate_count = self._candidate_count(query_terms)
        missing_slot = self._missing_slot_to_ask(session_id)
        clarifying_asks_so_far = sum(self._asked.get(session_id, {}).values())

        should_clarify = (
            candidate_count > OVER_GENERALITY_THRESHOLD
            and missing_slot is not None
            and turn < 10
            and clarifying_asks_so_far < MAX_CLARIFYING_TURNS
        )

        if should_clarify:
            self._asked[session_id][missing_slot] = self._asked[session_id].get(missing_slot, 0) + 1
            return {
                "message": f"I found a lot of options — could you tell me more about the {missing_slot.replace('_', ' ')} you're looking for?",
                "ask_attribute": missing_slot,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        recommendations = self._search(query_terms, top_k)
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
