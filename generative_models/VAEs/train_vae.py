import traceback
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import argparse
from data_loader import get_dataloaders, get_defect_dataset, get_no_defect_dataset, get_dataloader_by_class


class Vae(nn.Module):
    def __init__(self, latent_size=128, image_size=512):
        super(Vae, self).__init__()
        self.latent_size = latent_size

        # Encoder: four conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)

        # Compute flattened size dynamically
        s1 = compute_conv_output_size(image_size, 5, 2, padding=2)
        s2 = compute_conv_output_size(s1, 5, 2, padding=2)
        s3 = compute_conv_output_size(s2, 5, 2, padding=2)
        s4 = compute_conv_output_size(s3, 5, 2, padding=2)
        self.flattened_size = s4 * s4 * 128

        # Bottleneck
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.mu = nn.Linear(512, self.latent_size)
        self.logvar = nn.Linear(512, self.latent_size)

        # Decoder
        self.fc2 = nn.Linear(self.latent_size, 512)
        self.fc3 = nn.Linear(512, self.flattened_size)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def encoder(self, x):
        t = F.relu(self.conv1(x))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = t.view(x.size(0), -1)
        t = F.relu(self.fc1(t))
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def unflatten(self, x):
        size = int((self.flattened_size // 128) ** 0.5)
        return x.view(x.size(0), 128, size, size)

    def decoder(self, z):
        t = F.relu(self.fc2(z))
        t = F.relu(self.fc3(t))
        t = self.unflatten(t)
        t = F.relu(self.deconv1(t))
        t = F.relu(self.deconv2(t))
        t = F.relu(self.deconv3(t))
        t = torch.sigmoid(self.deconv4(t))
        return t

    def forward(self, x, y=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar



def compute_conv_output_size(input_size, kernel_size, stride, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def plot(epoch, recon, x):
    os.makedirs('images', exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(x[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Input')
        axes[1, i].imshow(recon[i, 0], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstruction')
    plt.tight_layout()
    plt.savefig(f'images/epoch_{epoch}.png')
    plt.close(fig)



def loss_function(x, recon, mu, logvar, kld_weight=1):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld_weight * kld
    return recon_loss, kld


def train(epoch, model, train_loader, optim, kld_weight=1):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i,(x,y) in enumerate(train_loader):
        try:
            optim.zero_grad()   
            pred, mu, logvar = model(x.to(device),y.to(device))
            
            recon_loss, kld = loss_function(x.to(device),pred, mu, logvar, kld_weight=kld_weight)
            loss = recon_loss + kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]
        except Exception as e:
            traceback.print_exe()
            torch.cuda.empty_cache()
            continue
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss

def test(epoch, model, test_loader, kld_weight=1):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                pred, mu, logvar = model(x.to(device),y.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar, kld_weight=kld_weight)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    plot(epoch, pred.cpu().data.numpy(), x.cpu().data.numpy())
            except Exception as e:
                traceback.print_exe()
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        



def load_data(data_dir, batch_size, num_workers, image_size):
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,  # Adjust validation split ratio if needed
        num_workers=num_workers,
        random_seed=42,  # Ensure reproducibility
        image_size=image_size,
    )

    return train_loader, test_loader

def load_data_by_class(file_paths, labels, batch_size, num_workers, image_size):
    train_loader, test_loader = get_dataloader_by_class(
        file_paths=file_paths,
        labels=labels,
        batch_size=batch_size,
        val_split=0.2,  # Adjust validation split ratio if needed
        num_workers=num_workers,
        random_seed=42,  # Ensure reproducibility
        image_size=image_size,
        )
    return train_loader, test_loader

def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--max_epoch", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--load_epoch", type=int, default=-1, help="Epoch to load for checkpoint (-1 for no checkpoint).")
    parser.add_argument("--latent_size", type=int, default=128, help="Size of the latent space.")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the input images.")
    parser.add_argument("--kld_weight", type=float, default=1, help="Weight for the KLD loss term.")
    parser.add_argument("--class_name", type=str, default="Defects", help="Class name for the dataset.")
    args = parser.parse_args()

    # Assign variables from parsed arguments
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_epoch = args.max_epoch
    device = torch.device(args.device)
    num_workers = args.num_workers
    load_epoch = args.load_epoch
    latent_size = args.latent_size
    image_size = args.image_size
    data_dir = args.data_dir
    kld_weight = args.kld_weight
    class_name = args.class_name
    image_size

    # Load data and initialize model
    if class_name == "Defects":
        file_paths, labels = get_defect_dataset(data_dir)
    elif class_name == "NoDefects":
        file_paths, labels = get_no_defect_dataset(data_dir)
    else:
        raise ValueError("Invalid class name. Use 'Defects' or 'NoDefects'.")
    
    train_loader, test_loader = load_data_by_class(
        file_paths=file_paths,
        labels=labels,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    print("Dataloader created for class: {}".format(class_name))
    model = Vae(latent_size=latent_size, image_size=image_size).to(device)
    print("Model created")

    # If the training is interrupted, you can load the model from the last checkpoint and continue training
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("Model {} loaded.".format(load_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch + 1, max_epoch):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer, kld_weight=kld_weight)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader, kld_weight=kld_weight)
        print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss: {}".format(i, max_epoch, train_total, train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss: {}".format(i, max_epoch, test_loss, test_kld, test_loss))

        if i % 50 == 0 or i == max_epoch - 1:
            print("Saving model...")
            save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))
    
