# TODimport os
import argparse
import numpy as np
import torch
from pytorch_fid import fid_score

def main():
    parser = argparse.ArgumentParser(description="Compute FID between two image folders.")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to real/original images.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to generated images.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    args = parser.parse_args()

    batch_size = 50
    dims = 2048

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    fid = fid_score.calculate_fid_given_paths(
        [args.real_dir, args.generated_dir],
        batch_size=batch_size,
        device=device,
        dims=dims
    )

    print(f"\n FID score: {fid:.4f}")

if __name__ == "__main__":
    main()
