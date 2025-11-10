"""
Create a seed database with pre-enrolled users.

To add a new user:
1. Create a directory under seed_db/audio_samples/ with the username (e.g., seed_db/audio_samples/john/)
2. Add 3 audio samples named sample_1.wav, sample_2.wav, sample_3.wav
3. Add the username to the SEED_USERS list below
4. Run from project root: python seed_db/seed_db.py
"""

import os
import sys

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import extract_embedding, load_audio_file
from src.database import save_multiple_embeddings


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# CONFIGURATION: Add users to enroll here
# ==========================================
SEED_USERS = [
    "phat",
    # Add more usernames here, e.g.:
    # "john",
    # "alice",
    # "bob",
]


def enroll_user(username: str) -> bool:
    """
    Enroll a single user from their audio samples.

    Args:
        username: The username to enroll

    Returns:
        True if successful, False otherwise
    """
    sample_dir = os.path.join(SCRIPT_DIR, "audio_samples", username)

    # Check if sample directory exists
    if not os.path.exists(sample_dir):
        print(f"❌ Directory not found: {sample_dir}")
        print(
            f"   Please create the directory and add 3 audio samples (sample_1.wav, sample_2.wav, sample_3.wav)"
        )
        return False

    # Check if all 3 samples exist
    sample_files = [
        os.path.join(sample_dir, "sample_1.wav"),
        os.path.join(sample_dir, "sample_2.wav"),
        os.path.join(sample_dir, "sample_3.wav"),
    ]

    for sample_file in sample_files:
        if not os.path.exists(sample_file):
            print(f"❌ Sample not found: {sample_file}")
            return False

    print(f"\n{'='*60}")
    print(f"Enrolling user: {username}")
    print(f"{'='*60}")
    print(f"✅ Found all 3 samples in {sample_dir}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = []
    audio_lengths = []

    for i, sample_file in enumerate(sample_files, 1):
        print(f"  Processing sample {i}: {sample_file}")
        try:
            sr, wav_np = load_audio_file(sample_file)
            audio_tuple = (sr, wav_np)
            embedding = extract_embedding(audio_tuple)
            embeddings.append(embedding)

            # Calculate audio length in seconds
            length_sec = len(wav_np) / sr
            audio_lengths.append(length_sec)
            print(f"    ✅ Shape: {embedding.shape}, Length: {length_sec:.2f}s")
        except Exception as e:
            print(f"    ❌ Error processing sample: {e}")
            return False

    # Save to database
    print(f"\n  Saving to database...")
    try:
        save_multiple_embeddings(username, embeddings, audio_lengths, sample_files)
        print(
            f"  ✅ User '{username}' enrolled successfully with {len(embeddings)} samples"
        )
        return True
    except Exception as e:
        print(f"  ❌ Error saving to database: {e}")
        return False


def main():
    """Main function to enroll seed users (assumes database is already initialized)"""
    # Database path in project root (parent directory)
    project_root = os.path.dirname(SCRIPT_DIR)
    db_path = os.path.join(project_root, "voice_auth.db")

    print("=" * 60)
    print("SEED DATABASE - ENROLLING USERS")
    print("=" * 60)
    print(f"Users to enroll: {', '.join(SEED_USERS)}")
    print(f"Database location: {db_path}")
    print()

    # Enroll all users
    success_count = 0
    fail_count = 0

    for username in SEED_USERS:
        if enroll_user(username):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print()
    print("=" * 60)
    print("ENROLLMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully enrolled: {success_count} user(s)")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count} user(s)")
    print(f"\nDatabase: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
