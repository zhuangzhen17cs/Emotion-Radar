### `utils/face_emotion.py`

# ```python
from deepface import DeepFace

def detect_face_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return None