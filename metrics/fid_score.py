import os
import argparse
import numpy as np
import torch
from pytorch_fid import fid_score
import shutil
import tempfile
from PIL import Image

def select_class_images(real_dir, synthetic_dir, class_name):
    """
    Selects the correct sub-folder for real images and filters synthetic images by class.
    Resizes all selected images to 299x299 and saves them in temporary folders.

    Args:
        real_dir (str): Path to the real images root folder.
        synthetic_dir (str): Path to the synthetic images folder.
        class_name (str): 'NoDefects' or 'Defects'.

    Returns:
        Tuple[str, str]: (path to temp real class folder, path to temp synthetic class folder)
    """
    # Map class name to class id
    class_map = {'NoDefects': '0', 'Defects': '1'}
    if class_name not in class_map:
        raise ValueError("class_name must be 'NoDefects' or 'Defects'")

    class_id = class_map[class_name]

    # Path to real images subfolder
    real_class_dir = os.path.join(real_dir, class_name)
    print(f"Real class directory: {real_class_dir}")
    if not os.path.isdir(real_class_dir):
        raise FileNotFoundError(f"Real class folder not found: {real_class_dir}")

    # Create temp dirs for resized real and synthetic images to avoid overwriting existing files
    temp_real_dir = tempfile.mkdtemp(prefix=f"real_{class_name}_resized_")
    temp_synth_dir = tempfile.mkdtemp(prefix=f"synthetic_{class_name}_resized_")

    # Resize and copy real images
    for fname in os.listdir(real_class_dir):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            src = os.path.join(real_class_dir, fname)
            dst = os.path.join(temp_real_dir, fname)
            try:
                with Image.open(src) as img:
                    img = img.resize((299, 299), Image.LANCZOS)
                    img.save(dst)
            except Exception as e:
                print(f"Could not process real image {fname}: {e}")

    # Filter, resize, and copy synthetic images by class id in filename (e.g., *_1.png for Defects)
    for fname in os.listdir(synthetic_dir):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            name, _ = os.path.splitext(fname)
            if name.endswith(f"_{class_id}"):
                src = os.path.join(synthetic_dir, fname)
                dst = os.path.join(temp_synth_dir, fname)
                try:
                    with Image.open(src) as img:
                        img = img.resize((299, 299), Image.LANCZOS)
                        img.save(dst)
                except Exception as e:
                    print(f"Could not process synthetic image {fname}: {e}")

    return temp_real_dir, temp_synth_dir

def compute_fid_score(real_dir, generated_dir, class_name, use_gpu=True, batch_size=4):
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

    # Take only the desired class images
    real_dir, generated_dir = select_class_images(real_dir, generated_dir, class_name)

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
    parser.add_argument("--class_name", type=str, choices=['NoDefects', 'Defects'], required=True, help="Class name to compute FID for.")
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
    fid = compute_fid_score(args.real_dir, args.generated_dir, args.class_name, args.use_gpu, args.batch_size)
    print(f"FID score: {fid:.4f}")
