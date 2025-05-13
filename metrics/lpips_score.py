import os
import argparse
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)  # Shape: [1, 3, H, W]

def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS between two image folders.")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to original images.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to generated images.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Inizializza modello LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(device)

    scores = []
    for filename in tqdm(os.listdir(args.real_dir)):
        path_real = os.path.join(args.real_dir, filename)
        path_gen = os.path.join(args.generated_dir, filename)

        if not os.path.exists(path_gen):
            continue

        img0 = load_and_preprocess(path_real).to(device)
        img1 = load_and_preprocess(path_gen).to(device)

        with torch.no_grad():
            dist = loss_fn(img0, img1).item()
        print(f"{filename}: {dist:.4f}")
        scores.append(dist)


    mean_lpips = np.mean(scores)
    std_lpips = np.std(scores)

    print("\nLPIPS Results")
    print(f"Mean: {mean_lpips:.4f}")
    print(f"Std:  {std_lpips:.4f}")

if __name__ == "__main__":
    main()
