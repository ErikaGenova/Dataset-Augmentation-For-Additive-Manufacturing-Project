import torch
import numpy as np
from torchvision.utils import save_image
from dcgan import Generator  # Ensure the Generator class is imported
import os

# Load the generator model
def load_generator(model_path, latent_dim, img_size, channels):
    generator = Generator()
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Set the model to evaluation mode
    return generator

# Generate images
def generate_images(generator, latent_dim, num_images, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    Tensor = torch.FloatTensor
    z = Tensor(np.random.normal(0, 1, (num_images, latent_dim)))
    gen_imgs = generator(z)
    for i, img in enumerate(gen_imgs):
        save_image(img, f"{output_dir}/image_{i}.png", normalize=True)
    print(f"Generated {num_images} images in {output_dir}")

# Example usage
if __name__ == "__main__":
    model_path = "saved_models/generator_epoch_199.pth"  # Path to the saved generator model
    latent_dim = 128
    img_size = 512
    channels = 1
    num_images = 10

    generator = load_generator(model_path, latent_dim, img_size, channels)
    generate_images(generator, latent_dim, num_images)