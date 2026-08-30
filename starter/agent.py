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
SLOT_VOCAB: dict[str, set[str]] = {
    "material": {
        "cotton", "polyester", "nylon", "leather", "wool",
        "spandex", "silk", "rayon", "fabric", "denim", "linen",
        "acrylic", "alloy", "aluminum", "brass", "canvas", "cashmere",
        "chiffon", "cubic zirconia", "eva", "faux fur", "faux leather",
        "fleece", "foam", "gemstone", "lace", "lycra", "mesh", "microfiber",
        "microfleece", "modal", "nubuck", "pearl", "polar fleece",
        "polycarbonate", "polyurethane", "pvc", "rubber", "satin",
        "stainless steel", "sterling silver", "straw", "suede", "synthetic",
        "terry", "viscose"
    },
    "color": {
        "black", "white", "blue", "red", "pink", "green", "brown",
        "gray", "grey", "purple", "yellow", "orange", "navy", "beige",
        "tan", "maroon", "olive", "cream", "gold", "golden", "silver",
        "rose", "bronze", "multicolor", "apricot", "camel", "charcoal",
        "clear", "fuchsia", "khaki", "peach", "rose gold", "transparent",
        "two-tone", "wine"
    },
    "size": {
        "xs", "xxs", "xxl", "xl", "small", "medium", "large",
        "s", "m", "l", "1x", "2x", "3x", "3xl", "4xl", "5xl", "6xl",
        "one size", "petite", "plus", "plus size", "regular", "short", "tall"
    },
    "style": {
        "casual", "formal", "athletic", "vintage", "slim", "loose",
        "oversized", "classic", "modern", "bohemian", "sporty",
        "hoop", "dangle", "stud", "chain", "pendant", "statement",
        "a-line", "bifold", "biketard", "bodycon", "bolero", "bootcut",
        "cargo", "choker", "clog", "crewneck", "drop", "espadrille",
        "gothic", "harajuku", "henley", "hoodie", "huggie", "jegging",
        "jogger", "kimono", "minimalist", "moccasin", "mule", "platform",
        "polo", "punk", "retro", "sheath", "shrug", "slide", "tunic",
        "turtleneck", "v-neck", "western", "wrap", "y2k"
    },
    "use_case": {
        "wedding", "work", "office", "gym", "running", "hiking",
        "party", "travel", "outdoor", "everyday", "beach", "winter",
        "summer", "date", "interview", "bachelorette party", "ballet",
        "basketball", "bedtime", "camping", "carnival", "christmas", 
        "cocktail", "cosplay", "cycling", "dance", "driving", "fishing",
        "golf", "graduation", "halloween", "lounge", "medical",
        "mother's day", "pilates", "pool", "prom", "rave", "school", 
        "skateboarding", "sleep", "soccer", "swimming", "tennis",
        "valentine's day", "yoga"
    },
}

FEATURE_DISCLOSURE_RE = re.compile(r"what matters is:\s*(.+)", re.IGNORECASE)
BRAND_RE = re.compile(r"\bby\s+([A-Z][a-zA-Z0-9&' -]{1,30})\b")

CLARIFY_PRIORITY = ['category', 'other', 'color', 'material', 'size', 'feature', 'style', 'use_case']
MAX_OTHER_ASKS = 4
MAX_CLARIFYING_TURNS = 4

# bm25() column weights: (parent_asin[unindexed, ignored], title, categories,
# features, details, store, description).
BM25_WEIGHTS = (0.0, 2.0, 12.0, 5.0, 4.0, 3.0, 0.0)


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

        # Tracks how many times we've already asked about each slot.
        self._asked: dict[str, dict[str, int]] = {}

        # Accumulates every term the user has typed across the session.
        self._all_terms: dict[str, list[str]] = {}

        # Pre-cached metadata dictionaries for rapid re-ranking.
        self._title_by_asin: dict[str, str] = {}
        self._text_by_asin: dict[str, str] = {}

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
                asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                self._title_by_asin[asin] = title
                full_text = " ".join((
                    title,
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                )).lower()
                self._text_by_asin[asin] = full_text
                batch.append(
                    (
                        asin,
                        title,
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
        and merges them into session state."""
        self._all_terms.setdefault(session_id, []).extend(_terms(user_message))

        lowered = f" {user_message.lower()} "
        tokens = set(_terms(user_message))
        constraints: dict[str, list[str]] = {}

        for slot, vocab in SLOT_VOCAB.items():
            token_hits = tokens & vocab
            phrase_hits = {v for v in vocab if " " in v and f" {v} " in lowered}
            hits = sorted(token_hits | phrase_hits)
            if hits:
                constraints[slot] = hits

        brand_match = BRAND_RE.search(user_message)
        if brand_match:
            constraints["brand"] = [brand_match.group(1).strip()]

        for slot, values in constraints.items():
            existing = self.history[session_id].setdefault(slot, [])
            for value in values:
                if value not in existing:
                    existing.append(value)

        feature_match = FEATURE_DISCLOSURE_RE.search(user_message)
        if feature_match:
            feature_slot = self.history[session_id].setdefault("feature", [])
            for chunk in feature_match.group(1).split(";"):
                chunk = chunk.strip(" .")
                if chunk and chunk not in feature_slot:
                    feature_slot.append(chunk)

        return constraints

    def handle_override(self, session_id: str, user_message: str) -> None:
        """Handles an intent-override message additively, updating conflicting slots."""
        new_text = extract_override_value(user_message)
        new_constraints = self.update_history(session_id, new_text)
        
        for slot, new_values in new_constraints.items():
            if new_values:
                self.history[session_id][slot] = new_values.copy()
                
        if not new_constraints:
            self.history[session_id]["feature"] = [new_text]

    def _detect_category(self, user_message: str) -> str | None:
        match = re.search(r"^I'm looking for (.*?)[.,]", user_message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------
    def _slot_terms(self, session_id: str) -> list[str]:
        terms: list[str] = []
        slots = self.history[session_id]
        for slot in ["category", "color", "material", "brand", "style", "use_case"]:
            for value in slots.get(slot, []):
                terms.extend(_terms(value))
        terms.extend(self._all_terms.get(session_id, []))
        return terms

    def _rerank_by_coverage(self, session_id: str, candidates: list[dict], top_k: int) -> list[dict]:
        slots = self.history[session_id]
        known_values = {
            v.lower()
            for slot in ("color", "material", "size", "style", "use_case", "brand")
            for v in slots.get(slot, []) if v
        }
        feature_terms = {t for v in slots.get("feature", []) for t in _terms(v)}

        if not known_values and not feature_terms:
            return candidates[:top_k]

        scored = []
        for rank, cand in enumerate(candidates):
            text = self._text_by_asin.get(cand["parent_asin"], "")
            coverage = sum(1 for v in known_values if v in text)
            coverage += sum(1 for t in feature_terms if t in text)
            scored.append((coverage, -rank, cand))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [cand for _, _, cand in scored[:top_k]]

    def _search(self, terms: list[str], top_k: int, category_terms: list[str] | None = None) -> list[dict]:
        FETCH_MULTIPLIER = 5
        unique_terms = list(dict.fromkeys(terms))[:60]
        if not unique_terms:
            return []

        fetch_k = top_k * FETCH_MULTIPLIER
        weights_sql = ", ".join(str(w) for w in BM25_WEIGHTS)

        if category_terms:
            unique_cat = list(dict.fromkeys(t for t in dict.fromkeys(category_terms) if t))[:8]
            other_terms = [t for t in unique_terms if t not in set(unique_cat)]
            if unique_cat and other_terms:
                cat_clause = " AND ".join(f'"{t}"' for t in unique_cat)
                other_clause = " OR ".join(f'"{t}"' for t in other_terms)
                expression = f"({cat_clause}) AND ({other_clause})"
            elif unique_cat:
                expression = " AND ".join(f'"{t}"' for t in unique_cat)
            else:
                expression = " OR ".join(f'"{t}"' for t in unique_terms)

            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {weights_sql}) LIMIT ?",
                (expression, fetch_k),
            ).fetchall()
            if rows:
                return [{"parent_asin": str(row[0])} for row in rows]

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {weights_sql}) LIMIT ?",
            (expression, fetch_k),
        ).fetchall()
        return [{"parent_asin": str(row[0])} for row in rows]

    def _missing_slot_to_ask(self, session_id: str) -> str | None:
        slots = self.history[session_id]
        asked = self._asked.setdefault(session_id, {})
        for slot in CLARIFY_PRIORITY:
            if slot == "other":
                if asked.get("other", 0) < MAX_OTHER_ASKS:
                    return "other"
                continue
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

        is_override = is_override_message(user_message)
        if is_override:
            self.handle_override(session_id=session_id, user_message=user_message)
        else:
            self.update_history(session_id, user_message)

        current_terms = _terms(user_message)
        slot_terms = self._slot_terms(session_id)
        query_terms = slot_terms + current_terms

        missing_slot = self._missing_slot_to_ask(session_id)
        clarifying_asks_so_far = sum(self._asked.get(session_id, {}).values())

        should_ask = (
            not is_override
            and turn <= 3 
            and missing_slot is not None
            and clarifying_asks_so_far < MAX_CLARIFYING_TURNS
        )

        category_terms = _terms(" ".join(self.history[session_id].get("category", [])))
        raw_candidates = self._search(query_terms, top_k, category_terms=category_terms)
        recommendations = self._rerank_by_coverage(session_id, raw_candidates, top_k)

        if should_ask:
            self._asked[session_id][missing_slot] = self._asked[session_id].get(missing_slot, 0) + 1
            if missing_slot == "other":
                ask_message = "Here are some options so far — and could you tell me more about what specifically matters to you?"
            else:
                ask_message = f"Here are some options so far — could you tell me more about the {missing_slot.replace('_', ' ')} you're looking for?"
            
            return {
                "message": ask_message,
                "ask_attribute": missing_slot,
                "recommendations": recommendations,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }