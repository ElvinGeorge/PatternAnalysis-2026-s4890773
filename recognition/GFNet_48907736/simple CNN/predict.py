"""
For single images at a time
python predict.py --weights cnn.pt image1.jpeg image2.jpeg
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from dataset import CLASSES
from modules import SimpleCNN

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--weights", default="best_cnn.pt")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

with torch.no_grad():
    for path in args.images:
        image = transform(Image.open(path)).unsqueeze(0).to(device)
        probs = torch.softmax(model(image), dim=1)[0]
        pred = probs.argmax().item()
        print(f"{path}: {CLASSES[pred]} (P(AD) = {probs[0]:.3f})")
