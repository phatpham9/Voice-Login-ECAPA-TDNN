"""
SQLite database module for voice authentication system.
Replaces JSON storage with proper database with audit trail and ACID properties.
"""

import sqlite3
import numpy as np
import pickle
from typing import List, Optional
from contextlib import contextmanager

DB_PATH = "voice_auth.db"


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
                audio_file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """
        )

        # Migration: Add audio_file_path column if it doesn't exist
        try:
            cursor.execute("SELECT audio_file_path FROM embeddings LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE embeddings ADD COLUMN audio_file_path TEXT")
            print("✅ Added audio_file_path column to embeddings table")

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


def save_multiple_embeddings(
    username: str,
    embeddings: List[np.ndarray],
    audio_lengths: Optional[List[float]] = None,
    audio_file_paths: Optional[List[str]] = None,
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
            audio_path = (
                audio_file_paths[idx - 1]
                if audio_file_paths and len(audio_file_paths) >= idx
                else None
            )

            cursor.execute(
                """INSERT INTO embeddings 
                (user_id, embedding, sample_number, audio_length_sec, audio_file_path) 
                VALUES (?, ?, ?, ?, ?)""",
                (user_id, emb_blob, idx, audio_len, audio_path),
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


def get_user_samples(username: str) -> Optional[List[dict]]:
    """Get detailed sample information for a user"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                e.id,
                e.sample_number,
                e.audio_length_sec,
                e.audio_file_path,
                e.created_at
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

        return [
            {
                "id": row[0],
                "sample_number": row[1],
                "audio_length_sec": row[2],
                "audio_file_path": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]


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
