# Emotion Radar AI

A real-time AI system that captures your **facial expressions** and **voice tone** to visualize your emotional state using a dynamic radar chart.

## 🔧 Features

- Real-time facial emotion recognition using webcam (powered by DeepFace)
- Voice emotion detection using MFCC features + ML classifier
- Live radar chart visualization (via Streamlit)
- Simple, intuitive web interface

## 🧪 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourname/emotion-radar-ai.git
cd emotion-radar-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
