"""
SQLite database module for voice authentication system.
Replaces JSON storage with proper database with audit trail and ACID properties.
"""

import sqlite3
import numpy as np
import pickle
import json
import os
from typing import List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "voice_auth.db"
OLD_JSON_PATH = "voice_db.json"


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize the SQLite database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Embeddings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                sample_number INTEGER NOT NULL,
                audio_length_sec REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """
        )

        # Authentication logs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                score REAL NOT NULL,
                threshold REAL NOT NULL,
                matched_sample INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_user_id 
            ON embeddings(user_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_auth_logs_username 
            ON auth_logs(username)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_auth_logs_timestamp 
            ON auth_logs(timestamp)
        """
        )

        print("✅ Database initialized successfully")


def migrate_from_json():
    """Migrate existing JSON data to SQLite database"""
    if not os.path.exists(OLD_JSON_PATH):
        print("ℹ️  No JSON file found - starting fresh")
        return

    try:
        with open(OLD_JSON_PATH, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if not json_data:
            print("ℹ️  JSON file is empty - nothing to migrate")
            return

        migrated_count = 0
        with get_db_connection() as conn:
            cursor = conn.cursor()

            for username, embeddings_data in json_data.items():
                # Check if user already exists
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                existing_user = cursor.fetchone()

                if existing_user:
                    print(f"⚠️  User '{username}' already exists - skipping")
                    continue

                # Insert user
                cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
                user_id = cursor.lastrowid

                # Handle both single embedding and multiple embeddings
                if isinstance(embeddings_data[0], list):
                    # Multiple embeddings
                    embeddings_list = embeddings_data
                else:
                    # Single embedding (backward compatibility)
                    embeddings_list = [embeddings_data]

                # Insert embeddings
                for idx, emb_list in enumerate(embeddings_list, start=1):
                    emb_array = np.array(emb_list, dtype="float32")
                    emb_blob = pickle.dumps(emb_array)

                    cursor.execute(
                        """INSERT INTO embeddings 
                        (user_id, embedding, sample_number) 
                        VALUES (?, ?, ?)""",
                        (user_id, emb_blob, idx),
                    )

                migrated_count += 1
                print(
                    f"✅ Migrated user '{username}' with {len(embeddings_list)} sample(s)"
                )

        print(f"\n🎉 Migration complete! Migrated {migrated_count} user(s)")

        # Create backup of JSON file
        backup_path = f"{OLD_JSON_PATH}.backup"
        os.rename(OLD_JSON_PATH, backup_path)
        print(f"📦 Original JSON backed up to: {backup_path}")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise


def save_multiple_embeddings(
    username: str,
    embeddings: List[np.ndarray],
    audio_lengths: Optional[List[float]] = None,
):
    """Save multiple embeddings for a user"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user:
            user_id = user[0]
            # Delete old embeddings
            cursor.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
            # Update timestamp
            cursor.execute(
                "UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,),
            )
        else:
            # Insert new user
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            user_id = cursor.lastrowid

        # Insert embeddings
        for idx, emb in enumerate(embeddings, start=1):
            emb_blob = pickle.dumps(emb)
            audio_len = (
                audio_lengths[idx - 1]
                if audio_lengths and len(audio_lengths) >= idx
                else None
            )

            cursor.execute(
                """INSERT INTO embeddings 
                (user_id, embedding, sample_number, audio_length_sec) 
                VALUES (?, ?, ?, ?)""",
                (user_id, emb_blob, idx, audio_len),
            )


def load_embedding(username: str) -> Optional[List[np.ndarray]]:
    """Load embedding(s) for a user - returns list of embeddings or None"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT e.embedding 
            FROM embeddings e
            JOIN users u ON e.user_id = u.id
            WHERE u.username = ?
            ORDER BY e.sample_number
        """,
            (username,),
        )

        rows = cursor.fetchall()

        if not rows:
            return None

        embeddings = []
        for row in rows:
            emb = pickle.loads(row[0])
            embeddings.append(emb)

        return embeddings


def list_users() -> List[str]:
    """List all enrolled users"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users ORDER BY username")
        rows = cursor.fetchall()
        return [row[0] for row in rows]


def delete_user(username: str) -> bool:
    """Delete a user and all their embeddings"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        return cursor.rowcount > 0


def get_user_info(username: str) -> Optional[dict]:
    """Get detailed information about a user"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                u.username,
                u.created_at,
                u.updated_at,
                COUNT(e.id) as sample_count
            FROM users u
            LEFT JOIN embeddings e ON u.id = e.user_id
            WHERE u.username = ?
            GROUP BY u.id
        """,
            (username,),
        )

        row = cursor.fetchone()

        if not row:
            return None

        return {
            "username": row[0],
            "created_at": row[1],
            "updated_at": row[2],
            "sample_count": row[3],
        }


def log_authentication(
    username: str,
    success: bool,
    score: float,
    threshold: float,
    matched_sample: Optional[int] = None,
):
    """Log an authentication attempt"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO auth_logs 
            (username, success, score, threshold, matched_sample)
            VALUES (?, ?, ?, ?, ?)
        """,
            (username, success, score, threshold, matched_sample),
        )


def get_auth_history(username: Optional[str] = None, limit: int = 50) -> List[dict]:
    """Get authentication history, optionally filtered by username"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if username:
            cursor.execute(
                """
                SELECT username, success, score, threshold, matched_sample, timestamp
                FROM auth_logs
                WHERE username = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (username, limit),
            )
        else:
            cursor.execute(
                """
                SELECT username, success, score, threshold, matched_sample, timestamp
                FROM auth_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = cursor.fetchall()

        return [
            {
                "username": row[0],
                "success": bool(row[1]),
                "score": row[2],
                "threshold": row[3],
                "matched_sample": row[4],
                "timestamp": row[5],
            }
            for row in rows
        ]


def get_database_stats() -> dict:
    """Get database statistics"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        # Total embeddings
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cursor.fetchone()[0]

        # Total auth attempts
        cursor.execute("SELECT COUNT(*) FROM auth_logs")
        total_attempts = cursor.fetchone()[0]

        # Successful attempts
        cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE success = 1")
        successful_attempts = cursor.fetchone()[0]

        # Recent activity (last 24 hours)
        cursor.execute(
            """
            SELECT COUNT(*) FROM auth_logs 
            WHERE timestamp > datetime('now', '-1 day')
        """
        )
        recent_attempts = cursor.fetchone()[0]

        return {
            "total_users": total_users,
            "total_embeddings": total_embeddings,
            "total_auth_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "failed_attempts": total_attempts - successful_attempts,
            "success_rate": (
                (successful_attempts / total_attempts * 100)
                if total_attempts > 0
                else 0
            ),
            "recent_attempts_24h": recent_attempts,
        }


# Initialize database on module import
init_database()

# Auto-migrate from JSON if it exists
if os.path.exists(OLD_JSON_PATH):
    print("\n🔄 Found existing JSON database - starting migration...")
    migrate_from_json()
