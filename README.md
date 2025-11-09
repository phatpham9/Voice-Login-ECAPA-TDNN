---
title: Voice Login with ECAPA-TDNN
emoji: 📚
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: mit
short_description: A text-independent speaker verification system.
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/677f2b5d19fb7f8c451e56e6/c5u7Lz_eGYRZeCDFem6v4.png
---

# 🔐 Voice Login with ECAPA-TDNN

A text-independent speaker verification system built with ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) from SpeechBrain. This application provides a complete voice-based authentication solution with enrollment, verification, and comprehensive performance analytics.

## 🎯 Overview

This system extracts unique voice characteristics (192-dimensional embeddings) from speech recordings to authenticate users. It's language-independent and doesn't require specific phrases, making it flexible and user-friendly.

**Key Capabilities:**
- Voice-based user enrollment with multi-sample support
- Real-time speaker verification and authentication
- SQLite database with full audit trail
- Interactive web interface powered by Gradio

## ✨ Features

### 🎤 Voice Login
- **Text-Independent**: Works with any spoken content - no fixed passphrases needed
- **Multi-Language Support**: Compatible with any language (Vietnamese, English, etc.)
- **Multiple Samples**: Enroll with 1-3 voice samples per user for improved accuracy
- **Best Match Algorithm**: Compares against all stored samples and uses the highest similarity score
- **Smart Audio Processing**: Automatic normalization (3-15 seconds) with quality feedback
- **Adjustable Threshold**: Configurable similarity threshold (default: 0.80) for security vs. usability balance

### 🗄️ Data Management
- **SQLite Database**: Robust storage with ACID properties and transaction support
- **Complete Audit Trail**: Logs every authentication attempt with timestamp and score
- **User Management**: Easy interface to view, update, and delete enrolled users
- **Statistics Dashboard**: System-wide statistics and recent activity monitoring
- **Auto-Migration**: Seamless upgrade from legacy JSON format

### 🎨 User Interface
- **Gradio Web Interface**: Clean, intuitive interface accessible via browser
- **Enrollment Tab**: Step-by-step voice sample collection with guided prompts
- **Login Tab**: Quick authentication with real-time feedback
- **Management Tab**: User administration and system statistics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/phatpham9/Voice-Login-ECAPA-TDNN.git
cd Voice-Login-ECAPA-TDNN

# Install dependencies with uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or use pip
pip install -r requirements.txt
```

### Running the Application

**Production Mode:**
```bash
uv run app.py

# Or
python app.py
```

**Development Mode (auto-reload on code changes):**
```bash
uv run gradio app.py

# Or
gradio app.py
```

The Gradio interface will launch in your browser at `http://localhost:7860`

## 📋 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 - 2.4.x
- **TorchAudio**: 2.0.0 - 2.4.x
- Gradio, SpeechBrain, NumPy, SoundFile

See `requirements.txt` for complete dependencies.

## 📖 How to Use

### Enrollment

1. Open the **Enroll** tab
2. Enter a unique username
3. Record 1-3 voice samples (3-10 seconds each):
   - Sample 1 (required): Natural speech, minimum 3 seconds
   - Samples 2-3 (optional): Additional samples improve accuracy
4. Click **Enroll** to register the voice profile

**Tips:**
- Speak naturally for 5+ seconds per sample
- Use different phrases for each sample
- Record in a quiet environment
- Each sample is stored separately for better matching

### Authentication

1. Open the **Login** tab
2. Enter your username
3. Record your voice (3-10 seconds of speech)
4. Optionally adjust the similarity threshold (default: 0.80)
5. Click **Login** to verify

The system compares your voice against all enrolled samples and uses the best match score.

- Retry enrollment with longer audio samples

### Threshold Tuning

- **0.85-0.95**: High security, may reject some legitimate users
- **0.80** (default): Balanced security and convenience
- **0.65-0.75**: More permissive, lower security

## 🏗️ Project Structure

```
Voice-Login-ECAPA-TDNN/
├── app.py                   # Main application entry point
├── src/                     # Source code modules
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Core utilities (model, audio processing, embeddings)
│   ├── database.py          # SQLite database operations
│   ├── ui_login.py          # Login tab UI
│   ├── ui_enroll.py         # Enrollment tab UI
│   ├── ui_manage.py         # User management tab UI
│   └── ui_statistics.py     # Statistics tab UI
├── requirements.txt         # Python dependencies
├── enrollment_texts.json    # Sample enrollment prompts
├── voice_auth.db            # SQLite database (auto-created)
├── ecapa/                   # Pre-trained model files (auto-downloaded)
└── README.md                # This file
```

**Module Overview:**
- **app.py**: Orchestrates all UI tabs and launches the Gradio interface
- **src/core.py**: ECAPA-TDNN model loading, audio processing, embedding extraction
- **src/database.py**: SQLite operations, user management, authentication logging
- **src/ui_*.py**: Individual UI tabs with their respective business logic

## 🔬 Technical Details

### Model Architecture
- **Model**: ECAPA-TDNN (SpeechBrain pre-trained)
- **Source**: `speechbrain/spkrec-ecapa-voxceleb`
- **Embedding Size**: 192 dimensions
- **Training Data**: VoxCeleb (1M+ utterances, 7000+ speakers)
- **Similarity Metric**: Cosine similarity

### Audio Processing
1. Convert to mono (if stereo)
2. Resample to 16kHz
3. Normalize length (3-15 seconds)
4. Extract 192D embedding via ECAPA-TDNN
5. Compare using cosine similarity

### Multi-Sample Strategy
- Each enrollment sample stored separately (not averaged)
- During verification, compares against all stored embeddings
- Uses maximum similarity score for robust matching
- More resilient to recording condition variations

## 💡 Best Practices

**For Better Accuracy:**
- Record 5-10 seconds of natural speech per sample
- Use 2-3 enrollment samples per user
- Maintain consistent recording conditions
- Ensure quiet environment with good microphone

**Troubleshooting:**
- Low scores? Check audio length (3+ seconds) and quality
- High false acceptance? Increase threshold (0.85-0.90)
- High false rejection? Lower threshold (0.70-0.75) or re-enroll with better samples

## 🎓 Academic Context

Developed as part of the **Advanced Image Processing** course at Saigon University (SGU), demonstrating practical applications of deep learning in biometric authentication and audio signal processing.

## 📚 References

- **ECAPA-TDNN Paper**: [Arxiv 2005.07143](https://arxiv.org/abs/2005.07143)
- **SpeechBrain**: [Documentation](https://speechbrain.readthedocs.io/)
- **Model**: [HuggingFace Model Card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **VoxCeleb Dataset**: [Official Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This is an educational project. For production deployment, implement additional security measures including liveness detection, encrypted storage, and multi-factor authentication.
