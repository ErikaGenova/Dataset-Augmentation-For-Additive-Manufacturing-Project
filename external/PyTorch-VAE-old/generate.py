import os
import torch
import argparse
from torchvision import transforms
import torchvision.utils as vutils
from models import VanillaVAE, ConditionalVAE
from dataset import DefectsDataset
import yaml
from torch.utils.data import DataLoader

def generate_images(checkpoint_path, config_path, data_dir, output_folder, model_name, device='cuda'):
    # Load config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the model
    model_params = config['model_params']

    if model_name == 'vanilla_vae':
        model = VanillaVAE(in_channels=model_params['in_channels'], latent_dim=model_params['latent_dim'])
    elif model_name == 'cvae':
        model = ConditionalVAE(
            in_channels=model_params['in_channels'],
            num_classes=model_params['num_classes'],
            latent_dim=model_params['latent_dim']
        )
    else:
        raise ValueError(f"Model {model} is not supported. Choose 'vanilla_vae' or 'cvae'.")
        
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

    if model_name == 'vanilla_vae':
        print("Generating images using Vanilla VAE...")
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
    elif model_name == 'cvae':
        num_samples = 30  # Number of samples to generate per class
        for class_label in range(model_params['num_classes']):
            # Create a folder for each class
            class_folder = os.path.join(output_folder, f"class_{class_label}")
            os.makedirs(class_folder, exist_ok=True)

            labels = torch.tensor([class_label] * num_samples, device=device)
            samples = model.sample(num_samples, current_device=device, labels=labels)

            # Save the generated samples
            for i, sample in enumerate(samples):
                output_path = os.path.join(class_folder, f"sample_{i}.png")
                vutils.save_image(sample.unsqueeze(0).data, output_path, normalize=True)
                print(f"Generated sample saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained VAE model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the VAE model checkpoint.")
    parser.add_argument('--config', type=str, required=True, help="Path to the VAE configuration file.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--output-folder', type=str, required=True, help="Path to the output folder for generated images.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for inference (e.g., 'cuda' or 'cpu').")
    parser.add_argument('--model', type=str, default='vanilla_vae', help="Model type to use for generation (e.g., 'vanilla_vae').")
    args = parser.parse_args()

    generate_images(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        output_folder=args.output_folder,
        model_name=args.model,
        device=args.device
    )