"""
GFNet. Takes a 224x224 slice and outputs either 0 = AD or 1 = CN.
"""
import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=14):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2) * 0.02)

    def forward(self, x):
        H, W = x.shape[1:3]
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            X = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
            X = X * torch.view_as_complex(self.complex_weight)
            x = torch.fft.irfft2(X, s=(H, W), dim=(1, 2), norm="ortho")
        return x.to(dtype)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.p
        return x * keep / (1 - self.p)


class GFBlock(nn.Module):
    def __init__(self, dim, h, w, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
    
    
class GFNet(nn.Module):
    def __init__(self, img_size=224, dim=256, depth=12, drop_path_rate=0.1):
        super().__init__()
        self.stem = PatchEmbed(dim)
        g = img_size // 16
        self.pos_embed = nn.Parameter(torch.zeros(1, g, g, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[GFBlock(dim, g, g, r) for r in rates])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.stem(x) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x).mean(dim=(1, 2))
        return self.head(x)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = GFNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model