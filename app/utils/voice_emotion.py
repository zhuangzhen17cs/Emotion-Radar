import librosa
import joblib

model = joblib.load("model/emotion_classifier.pkl")

def classify_audio_emotion(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = mfcc.mean(axis=1)
    emotion = model.predict([features])[0]
    return emotion
