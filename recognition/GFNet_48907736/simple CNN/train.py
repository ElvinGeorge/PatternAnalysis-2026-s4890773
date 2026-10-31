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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
best_val_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    for key, value in zip(history, [train_loss, val_loss, train_acc, val_acc]):
        history[key].append(value)
    print(f"Epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), args.save_path)
        print("New lowest val accuracy")

print(f"Best val accuracy: {best_val_acc:.3f}")

# Plots
epochs = range(1, len(history["train_loss"]) + 1)

plt.figure()
plt.plot(epochs, history["train_loss"], label="Training")
plt.plot(epochs, history["val_loss"], label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training and validation loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.plot(epochs, history["train_acc"], label="Training")
plt.plot(epochs, history["val_acc"], label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")

print("Saved loss_curve.png and accuracy_curve.png")