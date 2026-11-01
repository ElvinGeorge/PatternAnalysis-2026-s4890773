"""
python evaluate.py --data-root /content/ADNI_Processed --weights gfnet.pt
"""
import argparse
import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score)

from dataset import CLASSES, get_loader
from modules import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
parser.add_argument("--weights", default="gfnet.pt")
parser.add_argument("--split", default="test")  # test or val set
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args.weights, device)
loader = get_loader(args.data_root, args.split, 64)

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

# Confusion matrix
cm = confusion_matrix(labels, preds)
sensitivity = cm[0, 0] / cm[0].sum()  # AD correctly found
specificity = cm[1, 1] / cm[1].sum()  # CN correctly found

print(f"\n=== {args.split} set: {len(labels)} images ===")
print(classification_report(labels, preds, target_names=CLASSES, digits=3))
print(f"Accuracy:    {accuracy_score(labels, preds):.3f}")
print(f"Macro-F1:    {f1_score(labels, preds, average='macro'):.3f}")
print(f"Sensitivity: {sensitivity:.3f}  (AD recall)")
print(f"Specificity: {specificity:.3f}  (CN recall)")
print(f"AUROC:       {roc_auc_score(is_AD, prob_ad):.3f}")
print(f"Parameters:  {sum(p.numel() for p in model.parameters()):,}")
if device == "cuda":
    print(f"Peak VRAM:   {torch.cuda.max_memory_allocated() / 1e6:.1f} MB (batch of 64)")

# Inference time
image = next(iter(loader))[0][:1].to(device)
times = []
with torch.no_grad():
    for i in range(110):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(image)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 10:  # take median inference time of 100 runs excluding first 10
            times.append((time.perf_counter() - start) * 1000)
times.sort()
print(f"Median Inference Time:     {times[len(times) // 2]:.2f}ms per image")

# Plots
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues")
plt.title(f"Confusion matrix ({args.split})")
plt.savefig("GFNet_confusion_matrix.png")
plt.close()

RocCurveDisplay.from_predictions(is_AD, prob_ad, name="GFNet")
plt.title(f"ROC curve, AD as positive ({args.split})")
plt.savefig("GFNet_roc_curve.png")
plt.close()

# Save every prediction to a CSV
with open("GFNet_test_predictions.csv", "w") as file:
    file.write("path,true,pred,prob_AD,correct\n")
    for (path, _), y, p, prob in zip(loader.dataset.samples, labels, preds, prob_ad):
        file.write(f"{path},{CLASSES[y]},{CLASSES[p]},{prob:.3f},{int(y == p)}\n")

print("Done")