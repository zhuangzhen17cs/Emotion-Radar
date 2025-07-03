### `utils/face_emotion.py`

# ```python
from deepface import DeepFace

def detect_face_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # 提取主导情绪
        top_emotion = result[0]['dominant_emotion']

        # 提取完整概率分布，并统一为小写（与雷达图标签匹配）
        raw_probs = result[0]['emotion']
        prob_dict = {k.lower(): float(v)/100.0 for k, v in raw_probs.items()}

        return top_emotion.lower(), prob_dict

    except Exception as e:
        print("DeepFace error:", e)
        return None, None
