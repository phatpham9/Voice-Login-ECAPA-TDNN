# 🔐 Voice Login with ECAPA-TDNN

A text-independent speaker verification system built with ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) from SpeechBrain. This application provides a complete voice-based authentication solution with enrollment, verification, and comprehensive performance analytics.

## 🎯 Overview

This system extracts unique voice characteristics (192-dimensional embeddings) from speech recordings to authenticate users. It's language-independent and doesn't require specific phrases, making it flexible and user-friendly.

**Key Capabilities:**
- Voice-based user enrollment with multi-sample support
- Real-time speaker verification and authentication
- Performance metrics and analytics dashboard
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

### 📊 Performance Analytics
- **FAR/FRR Analysis**: False Acceptance and False Rejection Rate calculations
- **EER Calculation**: Automatic Equal Error Rate determination for optimal threshold tuning
- **ROC Curve**: Receiver Operating Characteristic with AUC scoring
- **DET Curve**: Detection Error Tradeoff visualization with log-scale plots
- **Score Distributions**: Histograms comparing genuine vs. impostor scores
- **Confusion Matrix**: Detailed breakdown of authentication results (TP/TN/FP/FN)
- **Interactive Threshold Testing**: Real-time metric updates as you adjust thresholds
- **Smart Recommendations**: Data-driven suggestions for system optimization

### 🗄️ Data Management
- **SQLite Database**: Robust storage with ACID properties and transaction support
- **Complete Audit Trail**: Logs every authentication attempt with timestamp and score
- **User Management**: Easy interface to view, update, and delete enrolled users
- **Statistics Dashboard**: System-wide metrics and recent activity monitoring
- **Auto-Migration**: Seamless upgrade from legacy JSON format

### 🎨 User Interface
- **Gradio Web Interface**: Clean, intuitive interface accessible via browser
- **Enrollment Tab**: Step-by-step voice sample collection with guided prompts
- **Login Tab**: Quick authentication with real-time feedback
- **Metrics Tab**: Interactive performance dashboard with charts and analysis
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
# or: python app.py
```

**Development Mode (auto-reload on code changes):**
```bash
uv run gradio app.py
# or: gradio app.py
```

The Gradio interface will launch in your browser at `http://localhost:7860`

## 📋 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 - 2.4.x
- **TorchAudio**: 2.0.0 - 2.4.x
- Gradio, SpeechBrain, NumPy, SoundFile
- Plotly 5.0+ (for metrics dashboard)
- SciPy 1.9+ (for EER calculation)

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

### Performance Analytics

Access the **Performance Metrics** tab to analyze system performance:

**Available Metrics:**
- **FAR/FRR**: False Acceptance and False Rejection Rates
- **EER**: Equal Error Rate (optimal threshold point)
- **ROC Curve**: True Positive vs False Positive rate
- **DET Curve**: Detection Error Tradeoff visualization
- **Score Distribution**: Genuine vs impostor score analysis
- **Confusion Matrix**: Detailed classification breakdown

**Requirements:**
- Minimum 10 genuine + 10 impostor attempts for meaningful analysis
- Recommended: 20+ attempts of each type

### Threshold Tuning

- **0.85-0.95**: High security, may reject some legitimate users
- **0.80** (default): Balanced security and convenience
- **0.65-0.75**: More permissive, lower security

Use the Performance Metrics tab to find the optimal threshold for your use case.

## 🏗️ Project Structure

```
Voice-Login-ECAPA-TDNN/
├── app.py                    # Main Gradio application
├── database.py               # SQLite database operations
├── performance_metrics.py    # Performance analysis & visualization
├── requirements.txt          # Python dependencies
├── enrollment_texts.json     # Sample enrollment prompts
├── voice_auth.db            # SQLite database (auto-created)
└── ecapa/                   # Pre-trained model files (auto-downloaded)
```

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
