import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
PROCESSED_DIR = r"E:/coding/3dcnn/project/data/processed"
CSV_PATH = r"E:/coding/3dcnn/project/Final_Table.csv"

# Ensure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Loading preprocessed data...")

# Load preprocessed frames and labels
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

# Load CSV for reference (optional, for debugging)
df = pd.read_csv(CSV_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"CSV shape: {df.shape}")

# Convert frame-level labels to video-level
# If any frame in the video has '1', mark the entire video as accident (1)
video_labels = np.array([1 if np.sum(labels) > 0 else 0 for labels in y])

# Count positives and negatives
pos_count = np.sum(video_labels == 1)
neg_count = np.sum(video_labels == 0)
print(f"Positive videos (Accidents): {pos_count}")
print(f"Negative videos (Normal): {neg_count}")

# Split data into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=video_labels
)

print("✅ Data split completed!")
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Save train and test datasets
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

# Also save video-level labels for reference
np.save(os.path.join(PROCESSED_DIR, "video_labels.npy"), video_labels)

print("All files saved successfully in 'processed' folder!")
