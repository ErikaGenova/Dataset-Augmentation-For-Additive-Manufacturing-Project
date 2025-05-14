import os
import argparse
import numpy as np
import torch
from pytorch_fid import fid_score

def fid_score(real_dir, generated_dir, use_gpu=True, batch_size=4):
    """
    Calculate the FID score between real and generated images.
    
    Args:
        real_dir (str): Path to the directory containing real images.
        generated_dir (str): Path to the directory containing generated images.
        use_gpu (bool): Whether to use GPU for computation.
        batch_size (int): Batch size for FID calculation.
    
    Returns:
        float: The FID score.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, generated_dir],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between two image folders.")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to real/original images.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to generated images.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for FID calculation.")
    args = parser.parse_args()

    # Check if the directories exist
    if not os.path.exists(args.real_dir):
        print(f"Real images directory does not exist: {args.real_dir}")
        exit(1)
    if not os.path.exists(args.generated_dir):
        print(f"Generated images directory does not exist: {args.generated_dir}")
        exit(1)
    # Check if the directories are empty
    if len(os.listdir(args.real_dir)) == 0:
        print(f"Real images directory is empty: {args.real_dir}")
        exit(1)
    if len(os.listdir(args.generated_dir)) == 0:
        print(f"Generated images directory is empty: {args.generated_dir}")
        exit(1)
    
    # Calculate FID score
    fid = fid_score(args.real_dir, args.generated_dir, args.use_gpu, args.batch_size)
    print(f"FID score: {fid:.4f}")
