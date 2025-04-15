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
import csv
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
from dataset import VideoDataset


class VideoTrainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, device, save_path='best_model.pth', log_path='training_log.csv'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.log_path = log_path
        self.best_acc = 0.0

        # Initialize logging file
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Acc'])

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for videos, labels in self.train_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                outputs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total
            val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(self.train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            # Logging
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, total_loss/len(self.train_loader), train_acc, val_acc])

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                print(f"Saved Best Model with Accuracy: {val_acc:.2f}%")

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in self.valid_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)


if __name__ == "__main__":

    args = parse_args()
    cfg = load_config(args, "/home/awd8324/onkar/SlowFast/configs/Kinetics/MVITv2_L_40x3_test.yaml")

    metadata = pd.read_csv(cfg.DATA.PATH_TO_DATA_DIR+'/Metadata.csv')
    train_metadata = metadata[metadata['train/valid'] == 'Train_video']
    valid_metadata = metadata[metadata['train/valid'] == 'Valid_video']

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((312, 312)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VideoDataset(train_metadata, cfg.DATA.PATH_TO_DATA_DIR, transform, num_frames=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)
    valid_dataset = VideoDataset(valid_metadata, cfg.DATA.PATH_TO_DATA_DIR, transform, num_frames=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(metadata['classname'].unique())
    model = build_model(cfg)
    check = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH)
    model.load_state_dict(check['model_state'], strict=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.head.projection.in_features  # Get the input features of the last layer
    model.head.projection = nn.Linear(in_features, num_classes)

    for param in model.head.projection.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # Define the output directory
    output_dir = "output"

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Update paths for saving model and logs
    save_path = os.path.join(output_dir, "best_model.pth")
    log_path = os.path.join(output_dir, "training_log.csv")

    trainer = VideoTrainer(model, train_loader, valid_loader, criterion, optimizer, device, save_path, log_path)
    trainer.train(num_epochs=cfg.TRAIN.NUM_EPOCHS)