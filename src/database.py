"""
SQLite database module for voice authentication system.
Replaces JSON storage with proper database with audit trail and ACID properties.
"""

import sqlite3
import numpy as np
import os
from typing import List, Optional
from contextlib import contextmanager

DB_PATH = "voice_auth.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """
    Convert numpy array to bytes using float32 format (more portable than pickle).

    Args:
        embedding: numpy array (typically 192-dimensional)

    Returns:
        bytes: Binary representation of the embedding
    """
    return embedding.astype(np.float32).tobytes()


def _bytes_to_embedding(data: bytes, dim: int = 192) -> np.ndarray:
    """
    Convert bytes back to numpy array.

    Args:
        data: Binary data
        dim: Expected dimension (default 192)

    Returns:
        numpy array
    """
    return np.frombuffer(data, dtype=np.float32).reshape(dim)


def init_database():
    """Initialize the SQLite database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

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

        # Embeddings table with constraints
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                sample_number INTEGER NOT NULL,
                audio_length_sec REAL CHECK(audio_length_sec IS NULL OR (audio_length_sec BETWEEN 1 AND 30)),
                audio_file_path TEXT,
                embedding_dim INTEGER DEFAULT 192,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, sample_number)
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


def save_multiple_embeddings(
    username: str,
    embeddings: List[np.ndarray],
    audio_lengths: Optional[List[float]] = None,
    audio_file_paths: Optional[List[str]] = None,
):
    """
    Save multiple embeddings for a user.
    Uses float32.tobytes() format instead of pickle for better portability.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user:
            user_id = user[0]
            # Get old audio file paths before deletion
            cursor.execute(
                "SELECT audio_file_path FROM embeddings WHERE user_id = ?", (user_id,)
            )
            old_paths = [row[0] for row in cursor.fetchall() if row[0]]

            # Delete old embeddings
            cursor.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))

            # Clean up old audio files
            _cleanup_audio_files(old_paths, username)

            # Update timestamp
            cursor.execute(
                "UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,),
            )
        else:
            # Insert new user
            cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
            user_id = cursor.lastrowid

        # Insert embeddings using bytes format
        for idx, emb in enumerate(embeddings, start=1):
            # Flatten the embedding if it's 2D
            if emb.ndim > 1:
                emb = emb.flatten()

            emb_blob = _embedding_to_bytes(emb)
            dim = emb.shape[0]  # Now this will be correct (e.g., 192)
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
                (user_id, embedding, sample_number, audio_length_sec, audio_file_path, embedding_dim) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, emb_blob, idx, audio_len, audio_path, dim),
            )


def load_embedding(username: str) -> Optional[List[np.ndarray]]:
    """
    Load embedding(s) for a user - returns list of embeddings or None.
    Uses float32 bytes format for portability.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT e.embedding, e.embedding_dim 
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
            emb_data = row[0]
            dim = row[1] if row[1] else 192  # Default to 192 if not set
            emb = _bytes_to_embedding(emb_data, dim)
            embeddings.append(emb)

        return embeddings


def _cleanup_audio_files(file_paths: List[str], username: str):
    """
    Clean up audio files associated with a user.
    Only deletes files in audio_samples/<username> directory to prevent accidental deletion.

    Args:
        file_paths: List of audio file paths to potentially delete
        username: Username for validation
    """
    if not file_paths:
        return

    audio_dir = f"audio_samples/{username}"
    deleted_count = 0

    for file_path in file_paths:
        if not file_path:
            continue

        # Safety check: only delete files in the user's audio_samples directory
        if file_path.startswith(audio_dir):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                print(f"⚠️  Warning: Could not delete audio file {file_path}: {e}")

    # Try to remove the user's directory if it's empty
    if deleted_count > 0:
        try:
            if os.path.exists(audio_dir) and not os.listdir(audio_dir):
                os.rmdir(audio_dir)
                print(f"🗑️  Removed empty directory: {audio_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove directory {audio_dir}: {e}")

    if deleted_count > 0:
        print(f"🗑️  Cleaned up {deleted_count} audio file(s)")


def list_users() -> List[str]:
    """List all enrolled users"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users ORDER BY username")
        rows = cursor.fetchall()
        return [row[0] for row in rows]


def delete_user(username: str) -> bool:
    """
    Delete a user and all their embeddings.
    Also cleans up associated audio files in audio_samples/<username> directory.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get audio file paths before deletion
        cursor.execute(
            """
            SELECT audio_file_path 
            FROM embeddings e
            JOIN users u ON e.user_id = u.id
            WHERE u.username = ?
            """,
            (username,),
        )
        audio_paths = [row[0] for row in cursor.fetchall() if row[0]]

        # Delete user (CASCADE will delete embeddings)
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        deleted = cursor.rowcount > 0

        # Clean up audio files
        if deleted and audio_paths:
            _cleanup_audio_files(audio_paths, username)

        return deleted


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
                e.embedding_dim,
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
                "embedding_dim": row[4],
                "created_at": row[5],
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
