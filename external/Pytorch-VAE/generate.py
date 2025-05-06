import torch
import numpy as np
import matplotlib.pyplot as plt
from train_cvae import Model, plot

def load_model(checkpoint_path, latent_size, image_size, device):
    """Load the model from a checkpoint."""
    model = Model(latent_size=latent_size, image_size=image_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

def generate_images(model, latent_size, num_images, device):
    """Generate new images using the trained model."""
    z = torch.randn(num_images, latent_size).to(device)  # Random latent vectors
    y = torch.tensor([i % model.num_classes for i in range(num_images)]).to(device)  # Alternate between classes
    label = np.zeros((num_images, model.num_classes))
    label[np.arange(num_images), y.cpu().numpy()] = 1
    label = torch.tensor(label).float().to(device)

    with torch.no_grad():
        generated_images = model.decoder(torch.cat((z, label), dim=1))
    return generated_images.cpu().numpy(), y.cpu().numpy()

def save_generated_images(images, labels, output_dir="generated_images"):
    """Save generated images to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.imsave(f"{output_dir}/image_{i}_class_{label}.png", image[0], cmap="gray")
    print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate images using a trained CVAE model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--latent_size", type=int, default=128, help="Size of the latent space.")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the input images.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for generation (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Directory to save generated images.")
    args = parser.parse_args()

    # Load the model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, args.latent_size, args.image_size, device)

    # Generate images
    images, labels = generate_images(model, args.latent_size, args.num_images, device)

    # Save images
    save_generated_images(images, labels, args.output_dir)