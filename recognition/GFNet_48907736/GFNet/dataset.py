"""
File structure is: ADNI_Processed/{train,val,test}/{AD,NC}/<patientID>_<slice>.jpeg

AD is 1 and NC is 0
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = {"NC": 0, "AD": 1}

def get_loader(data_root, split, batch_size=64):
    """Data set loader for the model"""
    # The MRI scans are made grayscale and resized to 224x224
    steps = [transforms.Grayscale(1), transforms.Resize((224, 224))]
    # Training images are augmented to prevent memorisation
    # Model was run with and without Affine transform and it was found to
    # significantly improve validation results. (See README.md)
    if split == "train":
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))]
    # Scale pixels from [0, 1] to [-1, 1] 
    steps += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

    data = datasets.ImageFolder(f"{data_root}/{split}", transform=transforms.Compose(steps))
    # Increase number of workers if running on an A100 to 4
    return DataLoader(data, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
