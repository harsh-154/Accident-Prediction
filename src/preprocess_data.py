import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# Paths
# =========================
DATA_DIR_ACCIDENT = r"E:/coding/3dcnn/project/data/CrashBest/output"
DATA_DIR_NORMAL = r"E:/coding/3dcnn/project/data/CrashBest/output_normal"
CSV_PATH = r"E:/coding/3dcnn/project/Final_Table.csv"
OUTPUT_DIR = r"E:/coding/3dcnn/project/data/processed"

IMG_SIZE = (112, 112)      # Resize frames
FRAMES_PER_VIDEO = 50      # Number of frames per video

# =========================
# Helper function to load frames
# =========================
def load_frames_from_folder(folder_path):
    frames = []
    frame_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    for file in frame_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img = cv2.resize(img, IMG_SIZE)
        frames.append(img)

    return np.array(frames)

# =========================
# Main preparation
# =========================
def prepare_data():
    df = pd.read_csv(CSV_PATH)
    print("CSV Columns:", df.columns.tolist())

    all_videos = []
    all_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_name = str(int(row['vidname']))

        # Check if folder exists in accident or normal directory
        folder_path = os.path.join(DATA_DIR_ACCIDENT, vid_name)
        if not os.path.exists(folder_path):
            folder_path = os.path.join(DATA_DIR_NORMAL, vid_name)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found!")
            continue

        # Load frames
        frames = load_frames_from_folder(folder_path)
        if len(frames) == 0:
            print(f"Skipping {vid_name}: No frames found")
            continue

        # Ensure exactly FRAMES_PER_VIDEO
        if frames.shape[0] >= FRAMES_PER_VIDEO:
            frames = frames[:FRAMES_PER_VIDEO]
        else:
            last_frame = frames[-1]
            pad_count = FRAMES_PER_VIDEO - frames.shape[0]
            frames = np.concatenate(
                [frames, np.repeat(last_frame[np.newaxis, ...], pad_count, axis=0)],
                axis=0
            )

        # Frame-level labels
        frame_labels = row.iloc[1:1 + FRAMES_PER_VIDEO].values.astype(int)

        all_videos.append(frames)
        all_labels.append(frame_labels)

    X = np.array(all_videos, dtype=np.uint8)  # (num_videos, 50, 112, 112, 3)
    y = np.array(all_labels, dtype=np.int32)  # (num_videos, 50)

    print(f"Final dataset shape: X={X.shape}, y={y.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("✅ Data saved successfully!")

if __name__ == "__main__":
    prepare_data()
