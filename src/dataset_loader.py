import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AccidentDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx] / 255.0  # normalize
        label = 1 if np.sum(self.labels[idx]) > 0 else 0  # accident if any frame has label 1

        video = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        label = torch.tensor(label, dtype=torch.long)

        return video, label

def get_dataloader(batch_size=2, split='train'):
    data_path = f"E:/coding/3dcnn/project/data/processed/X.npy"
    label_path = f"E:/coding/3dcnn/project/data/processed/y.npy"

    dataset = AccidentDataset(data_path, label_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
