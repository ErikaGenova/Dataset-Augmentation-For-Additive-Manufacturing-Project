import os
import torch
import argparse
from torchvision import transforms
import torchvision.utils as vutils
from models import VanillaVAE
from dataset import DefectsDataset
import yaml
from torch.utils.data import DataLoader

def generate_images(checkpoint_path, config_path, data_dir, output_folder, device='cuda'):
    # Load config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the model
    model_params = config['model_params']
    model = VanillaVAE(in_channels=model_params['in_channels'], latent_dim=model_params['latent_dim'])

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Transformations for the dataset
    val_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5830], std=[0.2075])
    ])

    # Load the dataset
    dataset = DefectsDataset(data_dir=data_dir, split='val', transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Generate images
    with torch.no_grad():
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            # Generate the reconstructed image
            reconstructed_img = model.generate(img)

            # Save the generated image
            output_path = os.path.join(output_folder, f"generated_{idx}.png")
            vutils.save_image(reconstructed_img.data, output_path, normalize=True)
            print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained VAE model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the VAE model checkpoint.")
    parser.add_argument('--config', type=str, required=True, help="Path to the VAE configuration file.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--output-folder', type=str, required=True, help="Path to the output folder for generated images.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for inference (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    generate_images(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        output_folder=args.output_folder,
        device=args.device
    )