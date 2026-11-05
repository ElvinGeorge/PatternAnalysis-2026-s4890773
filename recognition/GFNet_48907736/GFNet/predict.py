"""
Makes predictions using a given model for any given image

python predict.py --weights gfnet.pt image1.jpeg image2.jpeg
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from dataset import CLASSES
from modules import load_model

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--weights", default="gfnet.pt")
args = parser.parse_args()

# Use a GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args.weights, device)

# Same preprocessing as the val/test loader in dataset.py
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.CenterCrop(200),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Make predictions using the model
with torch.no_grad():
    for path in args.images:
        image = transform(Image.open(path)).unsqueeze(0).to(device)
        probs = torch.softmax(model(image), dim=1)[0]
        pred = probs.argmax().item()
        print(f"{path}: {CLASSES[pred]} (P(AD) = {probs[1]:.3f})")
