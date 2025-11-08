---
title: Voice Login ECAPA-TDNN
emoji: 📚
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: mit
short_description: Voice Login — ECAPA-TDNN (SpeechBrain)
---

# 🔐 Voice Login with ECAPA-TDNN

A text-independent speaker verification system using ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) from SpeechBrain. This application enables voice-based user enrollment and authentication.

## ✨ Features

- **Text-Independent**: Works with any spoken content, no specific phrases required
- **Multi-language Support**: Works with Vietnamese, English, and other languages
- **Pre-trained Model**: Uses SpeechBrain's pre-trained ECAPA-TDNN on VoxCeleb dataset
- **No Fine-tuning Required**: Ready to use out of the box
- **Simple Gradio Interface**: Easy-to-use web interface for enrollment and login
- **Adjustable Threshold**: Configurable similarity threshold for authentication (default: 0.80)
- **Multiple Sample Enrollment**: Support 1-3 voice samples per user for improved accuracy
- **Best Match Verification**: Compares against all enrolled samples and uses the highest score
- **Smart Audio Normalization**: Automatically handles audio length (3-15 seconds)
- **Audio Quality Warnings**: Provides feedback when recordings are too short for optimal accuracy

## 🎯 How It Works

The system uses ECAPA-TDNN to extract 192-dimensional speaker embeddings from voice recordings. These embeddings capture the unique characteristics of a person's voice and are used for:

1. **Enrollment**: Register a user by recording 1-3 voice samples. Each sample is stored as a separate embedding.
2. **Authentication**: Verify identity by comparing a new voice sample with ALL stored embeddings using cosine similarity, and using the best (maximum) score.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone phatpham9/Voice-Login-ECAPA-TDN
cd Voice-Login-ECAPA-TDNN

# Install dependencies with uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### Run the Application

#### Production Mode
```bash
# With uv
uv run app.py

# Or with python
python app.py
```

#### Development Mode (with auto-reload)
```bash
# With uv (recommended)
uv run gradio app.py

# Or with gradio CLI
gradio app.py
```

The application will launch a Gradio interface in your browser. In development mode, the server will automatically reload when you make changes to the code.

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0.0 - 2.4.x
- TorchAudio 2.0.0 - 2.4.x
- Gradio
- SpeechBrain
- NumPy
- SoundFile

See `requirements.txt` for full dependencies.

## 🎮 Usage

### Enrollment

1. Navigate to the **Enroll** tab
2. Enter a username
3. Record 1-3 voice samples:
   - **Sample 1 (Required)**: Record 3-10 seconds of speech
   - **Sample 2 (Optional)**: Record another 3-10 seconds with different content
   - **Sample 3 (Optional)**: Record a third sample for even better accuracy
4. Click "Enroll" to save the voice profile
5. Each sample is stored separately (not averaged)

**Tips for better enrollment:**
- Record at least 3-5 seconds per sample
- Use natural speech, not just a single word
- Vary your phrases across samples
- Avoid very short clips that require heavy padding

### Login

1. Navigate to the **Login** tab
2. Enter your username
3. Record 3-10 seconds of speech (or upload an audio file)
4. Adjust the similarity threshold if needed (default: 0.80)
5. Click "Login" to authenticate
6. The system compares your audio against ALL enrolled samples and uses the best match

**Login features:**
- Automatic audio length normalization (min 3s, max 15s)
- Warnings for audio that's too short
- Shows which enrolled sample matched best
- Displays similarity score and threshold

### Threshold Adjustment

- **Higher threshold (0.85-0.98)**: More secure but may reject legitimate users
- **Lower threshold (0.60-0.75)**: More permissive but less secure
- **Default (0.80)**: Balanced security and usability
- Adjust based on your security requirements and audio quality

### Understanding Results

**Successful Login:**
```
✅ SUCCESS — score=0.823 ≥ threshold=0.80 (matched sample 2/3)
```
This means your voice matched enrolled sample #2 with a score of 0.823.

**Failed Login:**
```
❌ DENIED — score=0.754 < threshold=0.80

⚠️ Warning: Audio is very short (1.2s). For better accuracy, record at least 3-5 seconds of speech.
```
The system provides helpful feedback on why verification failed.

## 🏗️ Architecture

```
Voice-Login-ECAPA-TDNN/
├── app.py              # Main Gradio application
├── db.py               # Simple JSON-based database for embeddings
├── requirements.txt    # Python dependencies
├── voice_db.json       # User embeddings storage (created at runtime)
└── ecapa/              # Pre-trained ECAPA-TDNN model files
    ├── classifier.ckpt
    ├── embedding_model.ckpt
    ├── hyperparams.yaml
    ├── label_encoder.ckpt
    └── mean_var_norm_emb.ckpt
```

## 🔬 Technical Details

### ECAPA-TDNN Model

- **Source**: `speechbrain/spkrec-ecapa-voxceleb`
- **Embedding Dimension**: 192D
- **Training Dataset**: VoxCeleb (1M+ utterances, 7k+ speakers)
- **Similarity Metric**: Cosine similarity

### Audio Processing Pipeline

1. **Input**: Audio file or microphone recording (any format supported by librosa)
2. **Conversion**: Convert to mono if stereo
3. **Resampling**: Resample to 16kHz (ECAPA-TDNN requirement)
4. **Normalization**: 
   - Minimum length: 3 seconds (padded with silence if shorter)
   - Maximum length: 15 seconds (trimmed if longer)
   - Optimal range: 3-10 seconds of actual speech
5. **Embedding Extraction**: ECAPA-TDNN generates 192D speaker embedding
6. **Storage/Comparison**: 
   - Enrollment: Store raw embeddings (up to 3 per user)
   - Login: Compare with all stored embeddings, use best match

### Multiple Sample Strategy

Instead of averaging embeddings (which can dilute unique characteristics), the system:
- Stores each enrollment sample as a separate embedding
- During verification, compares against ALL stored samples
- Uses the **maximum (best) similarity score**
- This approach is more robust to variations in recording conditions

### Audio Format Support

- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Channels**: Mono or Stereo (stereo is averaged to mono)
- **Formats**: WAV, MP3, M4A, FLAC, OGG, and more (via librosa/audioread)
- **Input Methods**: Microphone recording or file upload

## 🎯 Best Practices

### For Optimal Accuracy:

1. **Recording Length**: 
   - Aim for 5-10 seconds of continuous speech
   - Avoid very short clips (< 2 seconds)
   - Don't exceed 15 seconds (will be trimmed)

2. **Recording Quality**:
   - Use a good microphone in a quiet environment
   - Speak naturally at normal volume
   - Avoid background noise and echo

3. **Enrollment Strategy**:
   - Record 2-3 samples if possible
   - Use different phrases/sentences for each sample
   - Enroll and verify under similar conditions

4. **Threshold Selection**:
   - Start with default (0.80)
   - Lower (0.70-0.75) for convenience
   - Raise (0.85-0.90) for higher security

### Troubleshooting

**Low similarity scores?**
- Ensure recordings are long enough (3+ seconds)
- Check audio quality (no distortion/clipping)
- Try enrolling with longer samples
- Consider lowering the threshold

**Different users getting high scores?**
- Increase the threshold (0.85-0.90)
- Ensure enrollment samples are representative
- Check for audio quality issues

## 🎓 Educational Context

This project is part of the **Advanced Image Processing** course at Saigon University (SGU). It demonstrates practical applications of deep learning in biometric authentication and signal processing.

### Key Learning Outcomes:
- Speaker verification using deep learning
- Audio signal processing and feature extraction
- Biometric authentication systems
- Practical deployment with Gradio

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) for the pre-trained ECAPA-TDNN model
- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset for model training
- [Gradio](https://gradio.app/) for the web interface framework

## 📚 References

- [ECAPA-TDNN Paper](https://arxiv.org/abs/2005.07143)
- [SpeechBrain Documentation](https://speechbrain.readthedocs.io/)
- [Speaker Recognition on HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📧 Contact

For questions or feedback, please contact the course instructor or create an issue in the repository.

---

**Note**: This is an educational project for demonstration purposes. For production use, consider additional security measures such as liveness detection, secure storage, and multi-factor authentication.
