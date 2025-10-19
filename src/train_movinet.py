import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
PROCESSED_DIR = r"E:/coding/3dcnn/project/data/processed"
SEGMENTS_X = os.path.join(PROCESSED_DIR, "X_segments.npy")
SEGMENTS_Y = os.path.join(PROCESSED_DIR, "y_segments.npy")

# Choose a lightweight 3D model: MoViNet-A1 base from TF-Hub
# Expect input shape (T, H, W, C) with dynamic T. We'll use fixed T from segments
MOVINET_URL = "https://tfhub.dev/tensorflow/movinet/a1/base/kinetics-600/classification/3"

BATCH_SIZE = 4  # Reduced to avoid OOM
EPOCHS = 15
LEARNING_RATE = 1e-4


def create_dataset_generator(X_path: str, y_path: str, indices: np.ndarray, batch_size: int = BATCH_SIZE):
    """Create a generator that yields batches without loading full arrays into memory"""
    X_mmap = np.load(X_path, mmap_mode='r')
    y_mmap = np.load(y_path, mmap_mode='r')
    
    def generator():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_mmap[batch_indices].astype(np.float32) / 255.0
            y_batch = y_mmap[batch_indices].astype(np.int32)
            yield X_batch, y_batch
    
    return generator


def create_dataset(X_path: str, y_path: str, indices: np.ndarray) -> tf.data.Dataset:
    """Create dataset using generator to avoid memory issues"""
    generator = create_dataset_generator(X_path, y_path, indices)
    
    # Get shape info from first batch
    X_mmap = np.load(X_path, mmap_mode='r')
    sample_shape = X_mmap.shape[1:]  # (T, H, W, C)
    
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,) + sample_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_transfer_model(input_shape):
    # MoViNet expects dictionary input with 'image' key
    inputs = layers.Input(shape=input_shape, name='image')

    # MoViNet base without classification head
    backbone = hub.KerasLayer(MOVINET_URL, trainable=True, name="movinet_backbone")
    
    # Create dictionary input for MoViNet
    image_input = {'image': inputs}
    x = backbone(image_input)

    # The hub head outputs logits over Kinetics; we will add a custom projection.
    # Some MoViNet hub modules output feature dict; but this classification/3 variant outputs logits.
    # Use a small projection head for binary classification.
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model


def main():
    if not (os.path.exists(SEGMENTS_X) and os.path.exists(SEGMENTS_Y)):
        raise FileNotFoundError(
            "Segment files not found. Run src/segment_dataset.py to create X_segments.npy and y_segments.npy"
        )

    # Load only metadata to get shapes and labels for splitting
    X_mmap = np.load(SEGMENTS_X, mmap_mode='r')
    y_mmap = np.load(SEGMENTS_Y, mmap_mode='r')
    
    print(f"Dataset shape: {X_mmap.shape}")
    print(f"Labels shape: {y_mmap.shape}")
    
    # Create indices for train/val split without loading full arrays
    indices = np.arange(len(y_mmap))
    y_labels = y_mmap[:]  # Load only labels for stratification
    
    # Split indices instead of data
    train_indices, val_indices, _, _ = train_test_split(
        indices, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    train_ds = create_dataset(SEGMENTS_X, SEGMENTS_Y, train_indices)
    val_ds = create_dataset(SEGMENTS_X, SEGMENTS_Y, val_indices)

    timesteps, height, width, channels = X_mmap.shape[1:]
    model = build_transfer_model((timesteps, height, width, channels))

    # Check GPU availability
    print("Available devices:", tf.config.list_physical_devices())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    
    # Warmup: freeze backbone (simulate freezing layers) then train head
    model.get_layer('movinet_backbone').trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting warmup training (frozen backbone)...")
    model.fit(train_ds, validation_data=val_ds, epochs=max(2, EPOCHS // 3), verbose=1)

    # Fine-tune: unfreeze backbone for remaining epochs with small LR
    model.get_layer('movinet_backbone').trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting fine-tuning (unfrozen backbone)...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS - max(2, EPOCHS // 3), verbose=1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, 'best_model.h5')
    model.save(out_path)
    print(f"✅ Saved fine-tuned model to {out_path}")


if __name__ == "__main__":
    main()


