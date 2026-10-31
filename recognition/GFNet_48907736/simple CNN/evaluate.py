"""
Evaluates the saved model on the test set
python evaluate.py --data-root "C:/Users/elvin/OneDrive/Desktop/UQ/Year 3, Semester 2/COMP3710/Final Project/ADNI_Processed" --weights cnn.pt
"""
import argparse
import time
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score)
from dataset import CLASSES, get_loader
from modules import SimpleCNN

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--weights", default="cnn.pt")
parser.add_argument("--split", default="test") # test or val set
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

loader = get_loader(args.data_root, args.split)

# Predict all images
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

labels, preds, prob_ad = [], [], []
with torch.no_grad():
    for images, y in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        labels += y.tolist()
        preds += probs.argmax(1).tolist()
        prob_ad += probs[:, 0].tolist()

is_AD = [int(y == 0) for y in labels]