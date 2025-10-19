import torch
import numpy as np
import cv2
import os
from model_c3d import C3D

MODEL_PATH = "E:/coding/3dcnn/project/models/c3d_model.pth"

def predict(video_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3D(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load frames
    frames = sorted(os.listdir(video_folder))
    processed_frames = []

    for f in frames:
        img = cv2.imread(os.path.join(video_folder, f))
        img = cv2.resize(img, (112, 112))
        processed_frames.append(img)

    video = np.array(processed_frames) / 255.0
    video_tensor = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(video_tensor)
        _, predicted = torch.max(output, 1)

    return "Accident" if predicted.item() == 1 else "No Accident"

if __name__ == "__main__":
    result = predict("E:/coding/3dcnn/project/data/CrashBest/output/folder_1")
    print("Prediction:", result)
