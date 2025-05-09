import traceback
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import sys
import argparse
from data_loader import get_dataloaders


class Cvae(nn.Module):
    def __init__(self, latent_size=128, image_size=512, num_classes=2):
        super(Cvae,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes

        # For encode
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)

        # Dynamically compute the flattened size after convolutions
        conv1_output_size = compute_conv_output_size(image_size, kernel_size=5, stride=2)
        conv2_output_size = compute_conv_output_size(conv1_output_size, kernel_size=5, stride=2)
        self.flattened_size = conv2_output_size * conv2_output_size * 32

        self.linear1 = nn.Linear(self.flattened_size,300)
        self.mu = nn.Linear(300, self.latent_size)
        self.logvar = nn.Linear(300, self.latent_size)

        # For decoder
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300,self.flattened_size)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(1, 1, kernel_size=4)

    def encoder(self,x,y):
        # Class conditioning
        y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        y = torch.ones(x.shape).to(device)*y
        t = torch.cat((x,y),dim=1)
        # torch.Size([4, 2, 1024, 1280])
        
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        # print("t shape before reshape: ", t.shape)      # torch.Size([4, 32, 253, 317])
        t = t.reshape((x.shape[0], -1))
        # print("t shape after reshape: ", t.shape)       # torch.Size([4, 2566432])
        
        t = F.relu(self.linear1(t))     # linear1 is 512
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps*std + mu
    
    def unFlatten(self, x):
        # Compute the height and width dynamically
        conv2_output_size = int((self.flattened_size // 32) ** 0.5)
        return x.reshape((x.shape[0], 32, conv2_output_size, conv2_output_size))

    def decoder(self, z):
        t = F.relu(self.linear2(z))
        t = F.relu(self.linear3(t))
        t = self.unFlatten(t)
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = F.relu(self.conv5(t))
        return t


    def forward(self, x, y):
        mu, logvar = self.encoder(x,y)
        z = self.reparameterize(mu,logvar)

        # Class conditioning
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar
    
def compute_conv_output_size(input_size, kernel_size, stride, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def plot(epoch, pred, y,name='test_'):
    if not os.path.isdir('/content/cvae_images'):
        os.mkdir('/content/cvae_images')
    fig = plt.figure(figsize=(16,16))
    for i in range(len(pred)):
        ax = fig.add_subplot(3,2,i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
        ax.title.set_text(str(y[i]))
    plt.savefig("/content/cvae_images/{}epoch_{}.jpg".format(name, epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def loss_function(x, pred, mu, logvar, kld_weight=1):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld * kld_weight
    return recon_loss, kld


def train(epoch, model, train_loader, optim, kld_weight=1):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i,(x,y) in enumerate(train_loader):
        try:
            label = np.zeros((x.shape[0], 2))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)

            optim.zero_grad()   
            pred, mu, logvar = model(x.to(device),label.to(device))
            
            recon_loss, kld = loss_function(x.to(device),pred, mu, logvar, kld_weight=kld_weight)
            loss = recon_loss + kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]
            """
            if i == 0:
                print("Gradients")
                for name,param in model.named_parameters():
                    if "bias" in name:
                        print(name,param.grad[0],end=" ")
                    else:
                        print(name,param.grad[0,0],end=" ")
                    print()
            """
        except Exception as e:
            traceback.print_exc()
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
                label = np.zeros((x.shape[0], 2))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x.to(device),label.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar, kld_weight=kld_weight)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    # print("gr:", x[0,0,:5,:5])
                    # print("pred:", pred[0,0,:5,:5])
                    plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
            except Exception as e:
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        



def generate_image(epoch, z, y, model):
    with torch.no_grad():
        label = np.zeros((y.shape[0], 2))
        label[np.arange(z.shape[0]), y.cpu().numpy()] = 1  # Move `y` to CPU and convert to NumPy
        label = torch.tensor(label)

        pred = model.decoder(torch.cat((z.to(device), label.float().to(device)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().numpy(), name='Eval_')



def load_data(data_dir, batch_size, num_workers, image_size):
    # train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
    #                          transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
    #                          transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
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
    parser = argparse.ArgumentParser(description="Train a Conditional Variational Autoencoder (CVAE).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--max_epoch", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--load_epoch", type=int, default=-1, help="Epoch to load for checkpoint (-1 for no checkpoint).")
    parser.add_argument("--generate", type=bool, default=True, help="Whether to generate images during testing.")
    parser.add_argument("--latent_size", type=int, default=128, help="Size of the latent space.")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the input images.")
    parser.add_argument("--kld_weight", type=float, default=1, help="Weight for the KLD loss term.")
    args = parser.parse_args()

    # Assign variables from parsed arguments
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_epoch = args.max_epoch
    device = torch.device(args.device)
    num_workers = args.num_workers
    load_epoch = args.load_epoch
    generate = args.generate
    data_dir = args.data_dir
    latent_size = args.latent_size
    image_size = args.image_size
    kld_weight = args.kld_weight

    # Load data and initialize model
    train_loader, test_loader = load_data(data_dir, batch_size, num_workers, image_size)
    # print how many images are in each loader
    print("Train dataset size: ", len(train_loader.dataset))
    print("Test dataset size: ", len(test_loader.dataset))
    
    model = Cvae(latent_size, image_size).to(device)
    print("Model created.")
    
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("Model {} loaded".format(load_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    print("\n Starting training...")
    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch+1, max_epoch):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer, kld_weight=kld_weight)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader, kld_weight=kld_weight)
            if generate:
                z = torch.randn(6, latent_size).to(device)  # Generate random latent vectors
                y = torch.tensor([0, 1, 0, 1, 0, 1]).to(device)  # Alternate between class 0 and 1
                generate_image(i, z, y, model)
            
        print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss: {}".format(i, max_epoch,train_total, train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss: {}".format(i, max_epoch, test_loss, test_kld, test_loss))

        if i % 50 == 0 or i == max_epoch-1:
            # Save the model every 10 epochs
            print("Saving model...")
            save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))


    # i, (example_data, exaple_target) = next(enumerate(test_loader))
    # print(example_data[0,0].shape)
    # plt.figure(figsize=(5,5), dpi=100)
    # plt.imsave("example.jpg", example_data[0,0], cmap='gray',  dpi=1000)
    
