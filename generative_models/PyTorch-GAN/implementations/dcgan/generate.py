import torch
import numpy as np
from torchvision.utils import save_image
from dcgan import Generator  # Ensure the Generator class is imported
import os
import argparse

# Load the generator model
def load_generator(model_path, latent_dim, img_size, channels):
    generator = Generator(img_size, latent_dim, channels)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Set the model to evaluation mode
    return generator

# Generate images
def generate_images(generator, latent_dim, num_images, generate_defect, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    Tensor = torch.FloatTensor
    z = Tensor(np.random.normal(0, 1, (num_images, latent_dim)))
    gen_imgs = generator(z)
    class_label = 0 if generate_defect == "False" else 1
    for i, img in enumerate(gen_imgs):
        save_image(img, f"{output_dir}/image_{i}_class_{class_label}.png", normalize=True)
    print(f"Generated {num_images} images in {output_dir}")

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved generator model")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=512, help="Size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Directory to save generated images")
    parser.add_argument("--generate_defect", type=str, required=True, help="generate defect images") # "True" or "False"
    args = parser.parse_args()

    # Load the generator
    generator = load_generator(args.model_path, args.latent_dim, args.img_size, args.channels)

    # Generate images
    generate_images(generator, args.latent_dim, args.num_images, args.generate_defect, args.output_dir)

if __name__ == "__main__":
    main()