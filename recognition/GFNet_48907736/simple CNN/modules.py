"""
Simple CNN. Outputs either 0 = AD, 1 = CN.
"""
from torch import nn

def conv_block(in_ch, out_ch):
    # Conv -> BatchNorm -> ReLU -> MaxPool
    # Each block halves the spatial size
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    # Four conv blocks then a global average pooling and linear classifier
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Four conv blocks
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            # Global average pooling and linear classifier
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Returns logits
        return self.classifier(self.features(x))
