"""
File structure is: ADNI_Processed/{train,val,test}/{AD,CN}/<patientID>_<slice>.jpeg

AD is 0 and CN is 1
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["AD", "CN"]

def get_loader(data_root, split, batch_size=64):
    steps = [transforms.Grayscale(1), transforms.Resize((224, 224))]
    if split == "train":
        steps += [transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
 
    data = datasets.ImageFolder(f"{data_root}/{split}", transform=transforms.Compose(steps))
    return DataLoader(data, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
