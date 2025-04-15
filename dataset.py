import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import cv2
import random
from glob import glob
import numpy as np

# ========== Dataset ==========
class VideoDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None, num_frames=40, sampling_rate=3):
        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.label_map = {label: idx for idx, label in enumerate(metadata['classname'].unique())}

    def __len__(self):
        return len(self.metadata)

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def sample_frames(self, frames):
        total_frames = len(frames)
        clip_len = self.num_frames * self.sampling_rate
        if total_frames < clip_len:
            frames = frames + [frames[-1]] * (clip_len - total_frames)
        start = random.randint(0, max(0, total_frames - clip_len))
        sampled = [frames[start + i * self.sampling_rate] for i in range(self.num_frames)]
        return sampled

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_path = os.path.join(self.root_dir, row['train/valid'], row['video_name'])
        frames = self.read_video(video_path)
        frames = self.sample_frames(frames)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # Shape: (C, T, H, W)
        label = self.label_map[row['classname']]
        return frames, label