import librosa
import joblib
import numpy as np

model = joblib.load("model/emotion_classifier.pkl")
emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'fear', 'disgust', 'surprise']

def classify_audio_emotion(file_path):
    # 提取 MFCC 特征
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = mfcc.mean(axis=1).reshape(1, -1)

    # 主情绪预测
    top_emotion = model.predict(features)[0]

    # 如果模型支持概率输出
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]

        # 构建标签-概率字典
        model_labels = model.classes_
        prob_dict = {label.lower(): 0.0 for label in emotion_labels}
        for label, prob in zip(model_labels, probs):
            prob_dict[label.lower()] = float(prob)

        return top_emotion.lower(), prob_dict

    else:
        # 如果模型不支持概率输出，默认主情绪设为1，其余为0
        prob_dict = {label: 1.0 if label == top_emotion.lower() else 0.0 for label in emotion_labels}
        return top_emotion.lower(), prob_dict

