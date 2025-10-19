import numpy as np
import os

PROCESSED_DIR = r"E:/coding/3dcnn/project/data/processed"

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
video_labels = np.load(os.path.join(PROCESSED_DIR, "video_labels.npy"))

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)
print("Video-level labels shape:", video_labels.shape)
print("Sample video-level labels:", video_labels[:10])
