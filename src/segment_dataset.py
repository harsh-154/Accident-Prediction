import os
import math
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

# =========================
# Paths and constants
# =========================
DATA_DIR_ACCIDENT = r"E:/coding/3dcnn/project/data/CrashBest/output"
DATA_DIR_NORMAL = r"E:/coding/3dcnn/project/data/CrashBest/output_normal"

# CSV with per-frame labels; must include column 'vidname' followed by per-frame 0/1 labels
CSV_PATH = r"E:/coding/3dcnn/project/Final_Table.csv"

# Where to write segmented dataset arrays and metadata
OUTPUT_DIR = r"E:/coding/3dcnn/project/data/processed"

# Frame and segment config
IMG_SIZE = (112, 112)
SEGMENT_LENGTH = 16          # frames per segment (temporal depth)
SEGMENT_STRIDE = 8           # overlap stride; set to SEGMENT_LENGTH for non-overlap


def _read_frame_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'vidname' not in df.columns:
        raise ValueError("CSV must contain a 'vidname' column")
    return df


def _load_frames(folder_path: str) -> List[np.ndarray]:
    frame_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    frames: List[np.ndarray] = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        frames.append(img)
    return frames


def _segment_indices(num_frames: int, length: int, stride: int) -> List[Tuple[int, int]]:
    if num_frames == 0:
        return []
    indices: List[Tuple[int, int]] = []
    start = 0
    while start + length <= num_frames:
        indices.append((start, start + length))
        start += stride
    # ensure we cover the tail by adding a last window aligned to end if missed
    if len(indices) == 0 and num_frames < length:
        indices.append((0, num_frames))
    elif indices and indices[-1][1] < num_frames:
        end_aligned = num_frames
        start_aligned = max(0, end_aligned - length)
        if not indices or indices[-1] != (start_aligned, end_aligned):
            indices.append((start_aligned, end_aligned))
    return indices


def _label_for_window(frame_labels: np.ndarray, start: int, end: int) -> int:
    window = frame_labels[start:end]
    return int(np.any(window == 1))


def build_segment_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = _read_frame_labels(CSV_PATH)
    print("CSV Columns:", df.columns.tolist())

    segments: List[np.ndarray] = []
    segment_labels: List[int] = []
    metadata_rows: List[dict] = []

    accident_video_counter = 0
    normal_video_counter = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vidname = str(int(row['vidname'])) if not isinstance(row['vidname'], str) else row['vidname']

        # Resolve folder path between accident and normal roots
        folder_path = os.path.join(DATA_DIR_ACCIDENT, vidname)
        is_accident_video = True
        if not os.path.exists(folder_path):
            folder_path = os.path.join(DATA_DIR_NORMAL, vidname)
            is_accident_video = False

        if not os.path.exists(folder_path):
            print(f"Warning: frames folder not found for video {vidname}")
            continue

        frames_list = _load_frames(folder_path)
        if len(frames_list) == 0:
            print(f"Warning: no frames for {vidname}")
            continue

        num_frames = len(frames_list)
        frame_labels = row.iloc[1:].to_numpy(dtype=int)
        if frame_labels.shape[0] < num_frames:
            pad = np.zeros(num_frames - frame_labels.shape[0], dtype=int)
            frame_labels = np.concatenate([frame_labels, pad], axis=0)
        elif frame_labels.shape[0] > num_frames:
            frame_labels = frame_labels[:num_frames]

        # Decide base name prefix per video category
        if is_accident_video:
            accident_video_counter += 1
            video_prefix = f"VA{accident_video_counter}"
        else:
            normal_video_counter += 1
            video_prefix = f"VN{normal_video_counter}"

        # Windows
        windows = _segment_indices(num_frames, SEGMENT_LENGTH, SEGMENT_STRIDE)

        segment_index = 0
        for start, end in windows:
            segment_index += 1
            seg_frames = frames_list[start:end]

            # If last window shorter than length, pad last frame to length
            if len(seg_frames) < SEGMENT_LENGTH:
                last = seg_frames[-1]
                pad_count = SEGMENT_LENGTH - len(seg_frames)
                seg_frames = seg_frames + [last] * pad_count

            seg_array = np.stack(seg_frames, axis=0)  # (T,H,W,3)
            label = _label_for_window(frame_labels, start, end)

            segments.append(seg_array.astype(np.uint8))
            segment_labels.append(label)

            seg_name = f"{video_prefix}_{segment_index:03d}"
            metadata_rows.append({
                'segment_name': seg_name,
                'source_vidname': vidname,
                'video_prefix': video_prefix,
                'start_frame': start,
                'end_frame': min(end, num_frames),
                'label': label,
                'is_accident_video': int(is_accident_video)
            })

    if len(segments) == 0:
        raise RuntimeError("No segments created. Check paths and CSV schema.")

    X = np.stack(segments, axis=0)  # (N, T, H, W, 3)
    y = np.array(segment_labels, dtype=np.int32)  # (N,)

    print(f"Segments: X={X.shape}, y={y.shape}, positives={int(np.sum(y))}, negatives={int((y==0).sum())}")

    np.save(os.path.join(OUTPUT_DIR, "X_segments.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_segments.npy"), y)

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, "metadata_segments.csv"), index=False)
    print("✅ Saved X_segments.npy, y_segments.npy, metadata_segments.csv")


if __name__ == "__main__":
    build_segment_dataset()


