import json, os
import numpy as np

DB_PATH = "voice_db.json"


def _load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return {}


def _save_db(db: dict):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f)


def save_embedding(username: str, embedding: np.ndarray):
    """Save a single embedding (for backward compatibility)"""
    db = _load_db()
    db[username] = embedding.tolist()
    _save_db(db)


def save_multiple_embeddings(username: str, embeddings: list):
    """Save multiple embeddings for a user"""
    db = _load_db()
    # Store as list of embeddings
    db[username] = [emb.tolist() for emb in embeddings]
    _save_db(db)


def load_embedding(username: str):
    """Load embedding(s) for a user - returns list of embeddings or None"""
    db = _load_db()
    if username not in db:
        return None

    data = db[username]
    # Check if it's multiple embeddings (list of lists) or single (list of floats)
    if len(data) > 0 and isinstance(data[0], list):
        # Multiple embeddings
        return [np.array(emb, dtype="float32") for emb in data]
    else:
        # Single embedding (backward compatibility)
        return [np.array(data, dtype="float32")]


def list_users():
    db = _load_db()
    return list(db.keys())
