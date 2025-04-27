import os
import sys
import torch
from src.SinGAN.train import train_singan  # Importa il modulo di allenamento di SinGAN
from src.SinGAN.generate import generate_images  # Importa il modulo di generazione
from src.SinGAN.preprocess import preprocess_image  # Importa il modulo di preprocessamento

#TODO: fare in modo che il preprocessamento venga fatto in automatico per ogni immagine 

# Pre-processing function for input images
def generate_singan_images(input_image_path, output_dir,  num_iterations=10000):
    """
    Pre-process an input image and generate images using SinGAN.
    
    Parameters:
        input_image_path (str): Path to the input image.
        output_dir (str): Directory to save the generated images.
        num_iterations (int): Number of iterations for training SinGAN
    """
    # 1. Pre-process the input image
    preprocessed_image_path = os.path.join(output_dir, 'preprocessed_image.jpg')
    preprocess_image(input_image_path, preprocessed_image_path)

    # 2. Train SinGAN on the preprocessed image
    trained_model_dir = os.path.join(output_dir, 'trained_model')
    os.makedirs(trained_model_dir, exist_ok=True)

    # train SinGAN
    train_singan(preprocessed_image_path, trained_model_dir, num_iterations=num_iterations)

    # 3. Generate images using the trained SinGAN model
    generated_images_dir = os.path.join(output_dir, 'generated_images')
    os.makedirs(generated_images_dir, exist_ok=True)

    # generate images
    generate_images(trained_model_dir, generated_images_dir)

    print(f"Generated images saved to {generated_images_dir}")

# Perform the generation of images
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate images using SinGAN')
    parser.add_argument('--input_image_path', type=str, required=True, help='Path for the input image')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for generated images')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations for training SinGAN')
    args = parser.parse_args()

    generate_singan_images(args.input_image_path, args.output_dir, args.num_iterations)

