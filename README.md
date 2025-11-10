---
title: Voice Login with ECAPA-TDNN
emoji: 📚
colorFrom: gray
colorTo: purple
sdk: docker
app_port: 7860
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

### 🎤 Voice Authentication
- **Advanced Enrollment Strategy**: 
  - 3 mandatory voice samples (5-10 seconds each) for robust profile creation
  - Strict quality enforcement with automatic rejection of short samples
  - Vietnamese text verification prompts for anti-spoofing
- **Sophisticated Scoring System**:
  - Weighted fusion: 60% top-2 average + 40% centroid similarity
  - Dynamic centroid computed on-the-fly from all stored samples
  - Detailed score breakdown showing individual sample contributions
- **Audio Quality Diagnostics**:
  - Real-time analysis of amplitude, SNR, clipping, and noise floor
  - Actionable feedback to improve recording quality
  - Context-aware suggestions for enrollment and login
- **Anti-Spoofing Protection**:
  - Text verification using Whisper ASR (Vietnamese support)
  - Word Error Rate (WER) validation (threshold: 0.50)
  - Early rejection of replay attacks and synthetic voices
- **Multi-Language Support**: Compatible with Vietnamese, English, and other languages
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
- **Core Libraries**: 
  - SpeechBrain 1.0.3 (ECAPA-TDNN speaker verification)
  - OpenAI Whisper (ASR for text verification)
  - jiwer 4.0.0 (Word Error Rate calculation)
  - Gradio 4.16.0 (Web interface)
  - NumPy, SoundFile (Audio processing)

See `requirements.txt` for complete dependencies.

## 📖 How to Use

### Enrollment

1. Open the **Enroll** tab
2. Enter a unique username
3. Record **3 mandatory voice samples** (5-10 seconds each):
   - Read the displayed Vietnamese text prompt for each sample
   - All 3 samples must pass text verification (WER < 0.50)
   - Each sample must be at least 5 seconds long
   - System automatically validates audio quality
4. Click **Enroll** to register the voice profile

**Requirements:**
- **Minimum length**: 5 seconds per sample (strictly enforced)
- **Text verification**: Must read displayed prompts accurately
- **All samples mandatory**: Cannot enroll with fewer than 3 samples
- **Quality standards**: Clear audio with minimal noise and clipping

**Tips:**
- Read the Vietnamese text prompts clearly and naturally
- Record in a quiet environment with good microphone
- Speak at normal volume (avoid shouting or whispering)
- Wait for text prompt to refresh between samples
- Each sample is stored separately for robust matching

### Authentication

1. Open the **Login** tab
2. Enter your username
3. Read the displayed Vietnamese text prompt (refreshable)
4. Record your voice (5+ seconds recommended)
5. Optionally adjust the similarity threshold (default: 0.80)
6. Click **Login** to verify

**Two-Stage Verification:**
1. **Text Verification** (Anti-Spoofing): 
   - Whisper ASR transcribes your speech
   - WER calculated against expected text
   - Rejects if WER > 0.50 (likely replay attack or wrong text)
2. **Speaker Verification** (if text passed):
   - Compares voice against all enrolled samples
   - Uses weighted scoring: 60% top-2 average + 40% centroid
   - Shows detailed score breakdown with best match indicator

**Result Information:**
- Individual similarity scores for each stored sample
- Weighted scoring strategy breakdown
- Text verification details (expected, detected, WER)
- Audio quality diagnostics and improvement suggestions

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
├── whisper/                 # Whisper model files (auto-downloaded)
└── README.md                # This file
```

**Module Overview:**
- **app.py**: Orchestrates all UI tabs and launches the Gradio interface
- **src/core.py**: ECAPA-TDNN model loading, audio processing, embedding extraction, text verification
- **src/database.py**: SQLite operations, user management, authentication logging
- **src/ui_*.py**: Individual UI tabs with their respective business logic
- **enrollment_texts.json**: Vietnamese text prompts for anti-spoofing

## ⚙️ Configuration

Key parameters in `src/core.py`:

```python
# Enrollment Requirements
REQUIRED_ENROLLMENT_SAMPLES = 3    # Mandatory 3 samples
MIN_AUDIO_LENGTH_SEC = 5.0         # Minimum 5 seconds per sample

# Scoring Strategy (Weighted Fusion)
TOP_K_SAMPLES = 2                  # Use top 2 scores for averaging
SCORE_WEIGHT_TOP_K = 0.6          # 60% weight for top-k average
SCORE_WEIGHT_CENTROID = 0.4       # 40% weight for centroid similarity

# Anti-Spoofing
ENABLE_TEXT_VERIFICATION = True    # Enable/disable text verification
WER_THRESHOLD = 0.5                # Maximum Word Error Rate (50%)

# Authentication
DEFAULT_THRESHOLD = 0.80           # Default similarity threshold
```

## 🔬 Technical Details

### Model Architecture
- **Speaker Verification**: ECAPA-TDNN (SpeechBrain pre-trained)
  - Source: `speechbrain/spkrec-ecapa-voxceleb`
  - Embedding Size: 192 dimensions
  - Training Data: VoxCeleb (1M+ utterances, 7000+ speakers)
- **Speech Recognition**: Whisper-tiny (OpenAI)
  - Language: Vietnamese (`vi`)
  - Purpose: Anti-spoofing text verification
  - WER Threshold: 0.50

### Audio Processing Pipeline
1. Convert to mono (if stereo)
2. Resample to 16kHz
3. Quality analysis (amplitude, SNR, clipping, noise)
4. Text verification via Whisper ASR (if enabled)
5. Extract 192D embedding via ECAPA-TDNN
6. Compute weighted similarity score

### Enrollment Strategy
- **Mandatory samples**: Exactly 3 samples required
- **Minimum length**: 5 seconds per sample (strictly enforced)
- **Storage**: Each sample stored separately as 192D embedding (float32)
- **Text verification**: All samples must pass WER < 0.50 check
- **No averaging**: Preserves individual sample characteristics

### Scoring Strategy (Weighted Fusion)
- **Top-K Average** (60% weight): Average of top 2 similarity scores
  - Resilient to outliers and recording variations
  - Emphasizes best matches
- **Centroid Similarity** (40% weight): Compare against profile centroid
  - Centroid computed on-the-fly from all stored embeddings
  - Represents "average" voice characteristics
  - Not stored in database
- **Final Score**: `0.6 × top_k_avg + 0.4 × centroid_sim`
- **Similarity Metric**: Cosine similarity (range: -1 to 1, typically 0.5-1.0)

## 💡 Best Practices

**For Better Enrollment:**
- Read Vietnamese text prompts clearly and naturally
- Record 7-10 seconds per sample (exceeds 5s minimum)
- Maintain consistent recording conditions across all 3 samples
- Ensure quiet environment with good microphone
- Wait for quality diagnostics before proceeding
- If text verification fails, re-record and read prompt carefully

**For Better Login:**
- Read displayed text prompt accurately (check WER feedback)
- Speak naturally, similar to enrollment conditions
- Ensure minimum 5 seconds of speech
- Review diagnostic feedback if score is low
- Check text verification details if authentication fails

**Troubleshooting:**
- **Text verification fails**: Read prompt more carefully, check microphone clarity
- **Low similarity scores**: Check audio quality diagnostics, ensure 5+ seconds
- **High false acceptance**: Increase threshold (0.85-0.90)
- **High false rejection**: Lower threshold (0.70-0.75) or re-enroll with better samples
- **Replay attack detected**: Text verification working correctly (expected behavior)

## 🎓 Academic Context

Developed as part of the **Advanced Image Processing** course at Saigon University (SGU), demonstrating practical applications of deep learning in biometric authentication and audio signal processing.

## 🔐 Security Features

### Anti-Spoofing Protection
- **Text Verification**: Random Vietnamese prompts prevent replay attacks
- **Whisper ASR**: Transcribes speech and validates against expected text
- **WER Threshold**: 0.50 (50%+ accuracy required)
- **Early Rejection**: Text verification performed before speaker verification
- **Dynamic Prompts**: Refreshable texts prevent pre-recorded attacks

### Quality Assurance
- **Minimum Length**: 5 seconds enforced (prevents truncated attacks)
- **Audio Quality Analysis**: Real-time SNR, clipping, and noise detection
- **Diagnostic Feedback**: Actionable suggestions for improvement
- **Strict Enrollment**: All 3 samples must pass quality and text checks

### Database Security
- **SQLite with ACID**: Atomic transactions and data integrity
- **Audit Trail**: Complete logging of all authentication attempts
- **Embedding Storage**: Raw embeddings stored (not audio data)

## 📚 References

- **ECAPA-TDNN Paper**: [Arxiv 2005.07143](https://arxiv.org/abs/2005.07143)
- **SpeechBrain**: [Documentation](https://speechbrain.readthedocs.io/)
- **ECAPA Model**: [HuggingFace Model Card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Whisper ASR**: [OpenAI Whisper](https://github.com/openai/whisper)
- **VoxCeleb Dataset**: [Official Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This is an educational project demonstrating advanced voice authentication techniques. For production deployment, consider additional security measures including liveness detection, encrypted storage, multi-factor authentication, and regular security audits.
