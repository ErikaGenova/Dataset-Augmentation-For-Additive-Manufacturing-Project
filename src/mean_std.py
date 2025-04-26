import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Dataset and DataLoader that only convert images to tensors (no normalization)
dataset = datasets.ImageFolder(
    root='content/mla_project/images',
    transform=transforms.ToTensor()  # ToTensor() scales pixels to [0,1]
)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# 2. Initialize accumulators
sum_ = 0.0
sum_sq = 0.0
num_pixels = 0

for imgs, _ in loader:
    # imgs.shape = [Batch, Channels, Height, Width]
    B, C, H, W = imgs.shape
    # Count total number of pixels in this batch
    num_pixels += B * H * W

    # Sum all pixel values
    sum_ += imgs.sum()

    # Sum of squared pixel values
    sum_sq += (imgs ** 2).sum()

# 3. Compute mean and variance, then std
mean = sum_ / num_pixels
var = (sum_sq / num_pixels) - (mean ** 2)
std = torch.sqrt(var)

print(f"Mean grayscale: {mean.item():.4f}")
print(f"Std grayscale: {std.item():.4f}")
