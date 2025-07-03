import streamlit as st
import cv2
import tempfile
import time
from utils.face_emotion import detect_face_emotion  # expected to return (emotion, prob_dict)
from utils.audio_utils import record_audio
from utils.voice_emotion import classify_audio_emotion  # expected to return (emotion, prob_dict)
from utils.radar_plot import plot_emotion_radar

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Emotion Radar", layout="centered")
st.title("🎯 Emotion Radar AI Demo")
st.markdown("Analyze your **facial expression** and **voice emotion** in real time!")

# -------------------------------
# Session Initialization
# -------------------------------
if "face_emotion" not in st.session_state:
    st.session_state.face_emotion = None
    st.session_state.face_probs = None

if "voice_emotion" not in st.session_state:
    st.session_state.voice_emotion = None
    st.session_state.voice_probs = None

# -------------------------------
# 1️⃣ Facial Emotion Detection
# -------------------------------
# -------------------------------
# 1️⃣ Facial Emotion Detection (Button-based)
# -------------------------------
st.header("1️⃣ Facial Expression Detection")

if st.button("📷 Capture Frame"):
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", caption="Captured Frame")

        emotion, prob_dict = detect_face_emotion(frame_rgb)

        if emotion:
            st.session_state.face_emotion = emotion
            st.session_state.face_probs = prob_dict
        else:
            st.warning("No face detected.")
    else:
        st.error("Failed to capture from camera.")

# 显示结果（保留状态）
if st.session_state.face_emotion:
    st.success(f"🧠 Facial emotion: **{st.session_state.face_emotion}**")


# -------------------------------
# 2️⃣ Voice Emotion Detection
# -------------------------------
st.header("2️⃣ Voice Emotion Detection")

recording = st.button("🎤 Record Your Voice (4s)")
stop = st.button("🛑 Stop Recording")

if recording and not stop:
    with st.spinner("Recording..."):
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        record_audio(temp_audio, duration=4)

        emotion, prob_dict = classify_audio_emotion(temp_audio)
        st.session_state.voice_emotion = emotion
        st.session_state.voice_probs = prob_dict

if st.session_state.voice_emotion:
    st.success(f"🎙️ Voice emotion: **{st.session_state.voice_emotion}**")

# -------------------------------
# 📊 Radar Chart
# -------------------------------
st.header("📊 Emotion Radar Chart")

if st.button("📌 Show Radar Chart"):
    radar_labels = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]
    # 初始化全为 0 的分数
    emotion_scores = {label: 0.0 for label in radar_labels}

    # 加入面部情绪概率
    if st.session_state.face_probs:
        for k, v in st.session_state.face_probs.items():
            if k in emotion_scores:
                emotion_scores[k] += v

    # 加入语音情绪概率
    if st.session_state.voice_probs:
        for k, v in st.session_state.voice_probs.items():
            if k in emotion_scores:
                emotion_scores[k] += v

    # 归一化
    total = sum(emotion_scores.values())
    if total > 0:
        for k in emotion_scores:
            emotion_scores[k] /= total

    fig = plot_emotion_radar(emotion_scores, radar_labels)
    st.pyplot(fig)
