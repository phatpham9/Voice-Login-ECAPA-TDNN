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
    db = _load_db()
    db[username] = embedding.tolist()
    _save_db(db)


def load_embedding(username: str):
    db = _load_db()
    if username not in db:
        return None
    return np.array(db[username], dtype="float32")
