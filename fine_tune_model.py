import json

import os

import math

from tqdm import tqdm

from PIL import Image

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import get_model, get_model_weights


class CustomDataset(Dataset):

    def __init__(self, path: str, transform=None, train: bool = True):
        self.transform = transform
        self.path = path
        self.train = train
        self.mode = "train" if train else "test"

        with open(f"{path}/{self.mode}/{self.mode}.json", "r") as f:
            self.dataset_info = json.load(f)

        self.data = [(k, v) for k, v in self.dataset_info.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(f"{self.path}/{self.mode}/{image_path}")
        if self.transform:
            image = self.transform(image)
        return image, label
    
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

train_dataset = CustomDataset(path="./data/al5083/", transform=transform, train=True)

test_dataset = CustomDataset(path="./data/al5083/", transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Train dataset has {len(train_dataset)} images.")
print(f"Test dataset has {len(test_dataset)} images.")

# from sklearn.utils.class_weight import compute_class_weight
# 
# labels = [train_data[1] for train_data in train_dataset]
# class_weights = compute_class_weight(y=labels, classes=sorted(list(set(labels))), class_weight="balanced")
# print(class_weights)

# class_weights = [0.50745985, 2.49261544, 0.7026614, 1.10335981, 1.50502314, 1.57656379]

# Initialize checkpoint and weights
model_ckpt = "resnet18"
weight_ckpt = "IMAGENET1K_V1"

# Create directory for checkpoints if it doesn't exist
checkpoint_dir = "./data/weights/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device} as backend.")
device = torch.device(device)

# Define a custom model with dropout
class Model(nn.Module):

    def __init__(self, model_ckpt, weight_ckpt, num_classes, dropout_prob):
        super(Model, self).__init__()
        self.model = model = get_model(model_ckpt, weights=weight_ckpt)
        
        # Modify the first convolutional layer to accept grayscale (1 channel) images
        # The original conv1 layer has in_channels=3, we change it to 1
        self.model.conv1 = nn.Conv2d(in_channels=1, 
                                     out_channels=model.conv1.out_channels, 
                                     kernel_size=model.conv1.kernel_size, 
                                     stride=model.conv1.stride, 
                                     padding=model.conv1.padding, 
                                     bias=model.conv1.bias
                                    )
        # Replace the fully connected layer (fc) with a new one that includes dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),   # Add dropout before the final layer
            nn.Linear(self.model.fc.in_features, num_classes)  # Adjust for the number of output classes
        )

    def forward(self, x):
        return self.model(x)

# Load the custom model
model = Model(model_ckpt=model_ckpt, weight_ckpt=weight_ckpt, num_classes=6, dropout_prob=0.5)

# Define the loss function
class_weights = torch.Tensor([0.50745985, 2.49261544, 0.7026614, 1.10335981, 1.50502314, 1.57656379]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define optimizer with weight decay (L2 regularization)
lr = 1e-4
weight_decay = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Set number of epochs
num_epochs = 2
warmup_epochs = 3
end_epochs = 10

# Use a learning rate scheduler
def decay_learning_rate(epoch):
    if epoch < warmup_epochs:
        return 1
    if epoch > end_epochs:
        return 0.1
    epoch_ratio = (epoch - warmup_epochs) / (end_epochs - warmup_epochs)
    coeff = 0.5 * (1.0 + math.cos(math.pi * epoch_ratio))
    return 0.1 + coeff * 0.9
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_learning_rate)

# Initialize model
model = model.to(device)

# Check for existing checkpoints and load the most recent one
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoints found. Starting from scratch.")
    start_epoch = 0

# Training loop
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    # Print the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    # Print the average loss for this epoch
    print(f"Loss: {running_loss / len(train_loader):.4f}")

    # Validate the model on the test set
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Store labels and predictions for metric calculations
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate accuracy, precision, recall and f1 scores
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Print the accuracy, precision, recall and f1 score in percent
    print(f"Accuracy: {100*accuracy:.2f}%")
    print(f"Precision: {100*precision:.2f}%")
    print(f"Recall: {100*recall:.2f}%")
    print(f"F1 Score: {100*f1:.2f}%")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_ckpt}_epoch_{epoch + 1}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

# Initialize checkpoint dir
checkpoint_dir = "./data/weights"

# Get existing checkpoints
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
if checkpoints:
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Load precision, accuracy, recall and f1 scores
    precision_scores = [torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False)["precision"] for checkpoint_name in checkpoints]
    accuracy_scores = [torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False)["accuracy"] for checkpoint_name in checkpoints]
    recall_scores = [torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False)["recall"] for checkpoint_name in checkpoints]
    f1_scores = [torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False)["f1"] for checkpoint_name in checkpoints]
    # Create a DataFrame and print
    df = pd.DataFrame(
        {
            "checkpoint": checkpoints,
            "precision": precision_scores,
            "accuracy": accuracy_scores,
            "recall": recall_scores,
            "f1": f1_scores
        }
    )
    print(df)
else:
    print("No checkpoints found.")

