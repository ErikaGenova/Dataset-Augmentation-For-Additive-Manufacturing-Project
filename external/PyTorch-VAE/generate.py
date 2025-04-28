import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from dataset import DefectsDataset  # Importa il tuo dataset personalizzato
from models import VanillaVAE  # Importa il modello VAE

def generate_images(checkpoint_path, data_dir, output_folder, latent_dim=128, device='cuda'):
    # Carica il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Estrai il state_dict se il checkpoint è complesso (ad esempio, salvato con PyTorch Lightning)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Rimuovi il prefisso "model." dalle chiavi
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key[6:]] = value  # Rimuove "model."
        else:
            new_state_dict[key] = value

    # Carica il modello VAE
    model = VanillaVAE(in_channels=1, latent_dim=latent_dim)  # Modifica in_channels se necessario
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Trasformazioni per il dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Dimensioni coerenti con il modello
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
    ])

    # Carica il dataset
    dataset = DefectsDataset(data_dir=data_dir, split='val', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Crea la cartella di output
    os.makedirs(output_folder, exist_ok=True)

    # Genera immagini
    with torch.no_grad():
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            # Genera l'immagine ricostruita
            reconstructed_img = model.generate(img)

            # Salva l'immagine generata
            output_path = os.path.join(output_folder, f"generated_{idx}.png")
            reconstructed_img = reconstructed_img.squeeze(0).cpu()
            reconstructed_img = transforms.ToPILImage()(reconstructed_img)
            reconstructed_img.save(output_path)

            print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Generate images using a trained VAE model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the VAE model checkpoint.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--output-folder', type=str, required=True, help="Path to the output folder for generated images.")
    parser.add_argument('--latent-dim', type=int, default=20, help="Latent dimension of the VAE model.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for inference (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    # Genera immagini
    generate_images(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_folder=args.output_folder,
        latent_dim=args.latent_dim,
        device=args.device
    )