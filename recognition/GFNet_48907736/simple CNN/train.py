"""
python train.py --data-root "C:/Users/elvin/OneDrive/Desktop/UQ/Year 3, Semester 2/COMP3710/Final Project/ADNI_Processed" --epochs 20
"""
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import get_loader
from modules import SimpleCNN

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--save-path", default="cnn.pt")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

train_loader = get_loader(args.data_root, "train")
val_loader = get_loader(args.data_root, "val")

model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Each epoch
def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, count = 0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


# Go through epochs
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    for key, value in zip(history, [train_loss, val_loss, train_acc, val_acc]):
        history[key].append(value)
    print(f"Epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")

torch.save(model.state_dict(), args.save_path)
print(f"Saved final model to {args.save_path}")

# Loss and accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history["train_loss"], label="train")
ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss")
ax2.plot(history["train_acc"], label="train")
ax2.plot(history["val_acc"], label="val")
ax2.set_title("Accuracy")
for ax in (ax1, ax2):
    ax.set_xlabel("Epoch")
    ax.legend()