import os
import pandas as pd

data = []
base_path = "model\data\TESS Toronto emotional speech set data"

for actor_folder in os.listdir(base_path):
    emotion = actor_folder.split('_')[-1]
    folder_path = os.path.join(base_path, actor_folder)
    for fname in os.listdir(folder_path):
        if fname.endswith('.wav'):
            data.append({
                "filename": os.path.join(actor_folder, fname),
                "emotion": emotion
            })

df = pd.DataFrame(data)
df.to_csv("model/audio_labels.csv", index=False)
print("✅ 标签文件生成完毕: app/audio_labels.csv")
