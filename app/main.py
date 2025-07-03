import streamlit as st
import cv2
import tempfile
import time
from utils.face_emotion import detect_face_emotion
from utils.audio_utils import record_audio
from utils.voice_emotion import classify_audio_emotion
from utils.radar_plot import plot_emotion_radar

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="Emotion Radar", layout="centered")
st.title("🎯 Emotion Radar AI Demo")
st.markdown("Analyze your **facial expression** and **voice emotion** in real time!")

# -------------------------------
# Session State Initialization
# -------------------------------
if "face_emotion" not in st.session_state:
    st.session_state.face_emotion = None
if "voice_emotion" not in st.session_state:
    st.session_state.voice_emotion = None

# -------------------------------
# Face Emotion Recognition
# -------------------------------
st.header("1️⃣ Facial Expression Detection")

run_face = st.checkbox("Start Webcam", value=False)

FRAME_WINDOW = st.empty()
camera = cv2.VideoCapture(0)

if run_face:
    while run_face:
        _, frame = camera.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels='RGB')

        emotion = detect_face_emotion(frame_rgb)
        if emotion:
            st.session_state.face_emotion = emotion
            st.success(f"Detected facial emotion: **{emotion}**")
        else:
            st.warning("No face detected.")

        # Run at 1 FPS
        time.sleep(1)
        if not st.checkbox("Continue", value=True, key="face_continue"):
            break

camera.release()

# -------------------------------
# Voice Emotion Recognition
# -------------------------------
st.header("2️⃣ Voice Emotion Detection")

if st.button("🎤 Record Your Voice (4s)"):
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    record_audio(temp_audio, duration=4)
    voice_emotion = classify_audio_emotion(temp_audio)
    st.session_state.voice_emotion = voice_emotion
    st.success(f"Detected voice emotion: **{voice_emotion}**")

# -------------------------------
# Radar Chart
# -------------------------------
st.header("📊 Emotion Radar Chart")

if st.button("📌 Show Radar Chart"):
    radar_labels = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]

    # Fake scores for demo: only the detected emotions are emphasized
    emotion_scores = {label: 0.1 for label in radar_labels}
    if st.session_state.face_emotion:
        emotion_scores[st.session_state.face_emotion.lower()] += 0.5
    if st.session_state.voice_emotion:
        emotion_scores[st.session_state.voice_emotion.lower()] += 0.5

    fig = plot_emotion_radar(emotion_scores, radar_labels)
    st.pyplot(fig)
