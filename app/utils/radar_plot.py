import numpy as np
import matplotlib.pyplot as plt

def plot_emotion_radar(emotion_scores, labels):
    scores = [emotion_scores.get(label, 0) for label in labels]
    scores += scores[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={'polar': True})
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), labels)
    return fig
