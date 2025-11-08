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
- **Adjustable Threshold**: Configurable similarity threshold for authentication

## 🎯 How It Works

The system uses ECAPA-TDNN to extract 192-dimensional speaker embeddings from voice recordings. These embeddings capture the unique characteristics of a person's voice and are used for:

1. **Enrollment**: Register a user by recording their voice and storing the embedding
2. **Authentication**: Verify identity by comparing a new voice sample with the stored embedding using cosine similarity

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone phatpham9/Voice-Login-ECAPA-TDN
cd Voice-Login-ECAPA-TDNN

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will launch a Gradio interface in your browser.

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
3. Record approximately 5 seconds of audio (or upload an audio file)
4. Click "Enroll" to save the voice profile

### Login

1. Navigate to the **Login** tab
2. Enter your username
3. Record 3-5 seconds of audio (or upload an audio file)
4. Adjust the similarity threshold if needed (default: 0.85)
5. Click "Login" to authenticate

### Threshold Adjustment

- **Higher threshold (0.90-0.98)**: More secure but may reject legitimate users
- **Lower threshold (0.50-0.80)**: More permissive but less secure
- **Default (0.85)**: Balanced security and usability

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

### Audio Processing

- **Sample Rate**: 16kHz (automatically resampled)
- **Channels**: Mono (stereo is averaged)
- **Format**: Any format supported by TorchAudio/SoundFile

## 🎓 Educational Context

This project is part of the **Advanced Image Processing** course at Saigon University (SGU). It demonstrates practical applications of deep learning in biometric authentication and signal processing.

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