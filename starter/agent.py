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
    return "ignore" in lowered or "earlier preference" in lowered

def extract_override_value(message: str) -> str:
    """Extracts the new requirement from an override message."""
    if "what i need is:" in message.lower():
        return message.split(":", 1)[-1].strip(" .")
    return message

class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()

        # Create a storage for conversations and conversation history
        self.history: dict[str, dict[str, str]] = {}
        
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
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

        self.history[session_id] = {
            "category":[],
            'material':[],
            'color':[],
            'size':[],
            'style':[],
            'brand':[],
            'budget':[],
            'feature':[],
            'use_case':[],
            # 'other':[],
        }

    def handle_override(self, session_id: str, user_message: str) -> None:
        """Wipes old preferences while preserving the base category."""
        category = self.history[session_id].get("category", [])
        # Reset all slots
        self.history[session_id] = {slot: [] for slot in self.history[session_id]}
        # Restore category and append new requirement
        self.history[session_id]["category"] = category
        new_val = extract_override_value(user_message)
        self.history[session_id]["feature"].append(new_val)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        self.update_history(session_id, user_message)
        print(self.history[session_id])

        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]

        if turn == 1:
            match = re.search("^I'm looking for (.*?)[.,]", user_message)
            category = match.group(1).strip()
            self.history[session_id]['category'] = category
        if is_override_message(user_message):
            self.handle_override(session_id=session_id, user_message=user_message)

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


    def update_history(
      self,
      session_id,
      user_message      
    ):
        text = user_message.lower()
        constraints = {}
        
        materials = [
            "cotton", "polyester", "nylon", "leather",
            "wool", "spandex", "silk", "rayon", "fabric"
        ]

        colors = [
            "black", "white", "blue", "red", "pink",
            "green", "brown", "gray", "grey", "purple",
            "yellow", "orange"
        ]

        found_materials = [
            x for x in materials if x in text
        ]

        found_colors = [
            x for x in colors if x in text
        ]

        if found_materials:
            constraints["material"] = found_materials

        if found_colors:
            constraints["color"] = found_colors

        for slot, values in constraints.items():
            self.history[session_id][slot].extend(values)

        return constraints

