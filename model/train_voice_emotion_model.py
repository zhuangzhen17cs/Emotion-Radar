import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import librosa
import glob

# 假设你有音频文件和标签
AUDIO_DIR = "model\data\TESS Toronto emotional speech set data"  # 放你的训练音频
LABELS_FILE = "model/audio_labels.csv"  # 每个音频的标签，格式：filename,emotion

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# 加载数据
def load_dataset():
    import pandas as pd
    df = pd.read_csv(LABELS_FILE)
    features, labels = [], []

    for _, row in df.iterrows():
        file_path = os.path.join(AUDIO_DIR, row["filename"])
        if os.path.exists(file_path):
            feat = extract_features(file_path)
            features.append(feat)
            labels.append(row["emotion"])
    return np.array(features), np.array(labels)

# 训练模型
def train_and_save():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("模型评估结果：")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model/emotion_classifier.pkl")
    print("✅ 模型已保存到 model/emotion_classifier.pkl")

if __name__ == "__main__":
    train_and_save()
