import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

# Load data
PROCESSED_DIR = r"E:/coding/3dcnn/project/data/processed"

print("Loading data for training...")
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

# Convert frame-level to video-level labels
# If any frame in the video has a crash, label as crash (1)
y_train_video = np.array([1 if np.sum(label) > 0 else 0 for label in y_train])
y_test_video = np.array([1 if np.sum(label) > 0 else 0 for label in y_test])

print(f"Train set: {X_train.shape}, Labels: {y_train_video.shape}")
print(f"Test set: {X_test.shape}, Labels: {y_test_video.shape}")

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(50, 112, 112, 3)),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),

    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),

    Conv3D(128, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train_video,
    validation_data=(X_test, y_test_video),
    epochs=15,
    batch_size=8
)

# Save the trained model
model.save("E:/coding/3dcnn/project/accident_detection_3dcnn.h5")
print("Model saved successfully!")
