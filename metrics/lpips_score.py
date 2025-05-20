import os
import re
import sys
import glob
import argparse
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# get transformations to have size 
def get_transformations(args):
    # Estensioni dei file immagine da considerare
    image_extensions = ('*.png', '*.jpg', '*.jpeg')

    # Lista di tutti i file immagine
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(args.full_generated_dir, ext)))

    # Verifica se sono presenti immagini
    if not all_images:
        raise FileNotFoundError(f"No images found in {args.full_generated_dir}")

    # Prendi il primo file immagine
    first_image_path = all_images[0]

    # Ottieni le dimensioni (larghezza, altezza) dell'immagine
    with Image.open(first_image_path) as img:
        original_size = img.size  # (width, height)

    # Transforms for training and validation
    data_transforms =  transforms.Compose([
        transforms.Resize(original_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # standard normalization for grayscale
    ])

    return data_transforms


def load_and_preprocess(image_path, args):
    img = Image.open(image_path).convert('RGB')

    transform = args.data_transforms

    return transform(img).unsqueeze(0)  # Shape: [1, 3, H, W]

def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS between two image folders.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available.")
    parser.add_argument("--original_dir", type=str, required=True, help="Path to original images.", default="/content/images/original/")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to generated images.")
    parser.add_argument("--model", type=str, required=True, help="Model name (GANs, CVAE, VAE or Diffusion)")
    parser.add_argument("--experiment", type=str, required=True, help="Number of the experiment.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # verify if the model exists
    valid_models = ["CVAE", "VAE", "Diffusion", "GANs", "SinGAN"]

    if args.model not in valid_models:
        print(f"The model {args.model} doesn't exist. Choose either 'CVAE', 'VAE', 'Diffusion', 'GANs' or 'SinGAN'.")
        sys.exit()

    # Inizialized L-PIPS model
    loss_fn = lpips.LPIPS(net='alex').to(device)

    scores = []

    # generate the full path of generated images
    args.full_generated_dir = os.path.join(args.generated_dir, args.model, args.experiment)

    # verify if it exists or not
    if not os.path.exists(args.full_generated_dir):
        print(f"Doesn't exist the path of generated images: {args.full_generated_dir}")
        sys.exit()

    # define data_transforms
    args.data_transforms = get_transformations(args)

    # iterate on the generated images
    for filename in os.listdir(args.full_generated_dir):
        # verifiy the number of class in the name 
        match = re.search(r'class_(\d+)', filename)
        
        if match:
            class_id = int(match.group(1))
            if class_id == 0: # NoDefects
                full_original_dir = os.path.join(args.original_dir, "NoDefects")
            elif class_id == 1: #Defects
                full_original_dir = os.path.join(args.original_dir, "Defects")
            else:
                print("Don't recognize the class")
                sys.exit()
        else:
            print(f"filename {filename} hasn't the class number in the name")
            sys.exit()     

        # Doesn't exist the path of original directory
        if not os.path.exists(full_original_dir):
            print(f"Doesn't exist the path of original directory: {full_original_dir}")
            sys.exit()

        # create the full path of the generated image
        path_gen = os.path.join(args.full_generated_dir, filename)

        # Doesn't exist the path of generated image
        if not os.path.exists(path_gen):
            print(f"Doesn't exist the path of generated image: {path_gen}")
            sys.exit()
        
        print(f"\nImage: {path_gen}")
        
        # pre process generated image
        generated_image = load_and_preprocess(path_gen, args).to(device)

        min_dist = float('inf')  # inizializza il minimo

        # iterate on all original images
        for filename in tqdm(os.listdir(full_original_dir)):
            # create the full oath of the generated image
            path_original = os.path.join(full_original_dir, filename)

            # Doesn't exist the path of original image
            if not os.path.exists(path_original):
                print(f"Doesn't exist the path of generated image: {path_gen}")
                sys.exit()
            
            # pre process original image
            original_image = load_and_preprocess(path_original, args).to(device)

            with torch.no_grad():
                dist = loss_fn(original_image, generated_image).item()

            # if previous L-PIPS is higher than the current, I change it
            if dist < min_dist:
                min_dist = dist
        
        print(f"lpips: {min_dist}\n")
        scores.append(min_dist)

    mean_lpips = np.mean(scores)
    std_lpips = np.std(scores)

    print(f"\nLPIPS Results on the generated images from {args.experiment} of {args.model}")
    print(f"L-PIPS: {mean_lpips:.4f} ± {std_lpips:.4f}")

if __name__ == "__main__":
    main()
