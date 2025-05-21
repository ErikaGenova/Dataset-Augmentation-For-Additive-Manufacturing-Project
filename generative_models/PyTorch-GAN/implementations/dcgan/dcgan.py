import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from data_loader import get_all_data, get_defect_dataset, get_no_defect_dataset
from data_loader import DefectDataset

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        self.ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * self.ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class R1(nn.Module):
    def __init__(self, weight=10.0):
        super(R1, self).__init__()
        self.weight = weight

    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        grad_real = torch.autograd.grad(
            outputs=prediction_real.sum(),
            inputs=real_sample,
            create_graph=True
        )[0]
        reg_loss = self.weight * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return reg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="adam: learning rate for discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--data_dir", type=str, default="/content/mla_project/images/original/Defects", help="path to dataset")
    parser.add_argument("--generate_defect", type=str, default="True", help="generate defect images")
    parser.add_argument("--R1_regularization", type=str, default="False", help="use R1 regularization")
    parser.add_argument("--R1_lambda", type=float, default=10.0, help="lambda for R1 regularization")
    parser.add_argument("--output_dir", type=str, default=None, help="directory to save models")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False
    print("Using GPU" if cuda else "Using CPU")

        # Directory to save models
    if opt.output_dir is not None:
        output_dir = os.path.join(opt.output_dir, "saved_models")
        images_dir = os.path.join(opt.output_dir, "images")   
    else:
        output_dir = "saved_models"
        images_dir = "images"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Images directory: {images_dir}")

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(opt.img_size, opt.latent_dim, opt.channels)
    discriminator = Discriminator(opt.img_size, opt.channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    if opt.generate_defect == "True":
        print("Generating defect images...")
        file_paths, labels = get_defect_dataset(opt.data_dir)
    else:
        print("Generating no defect images...")
        file_paths, labels = get_no_defect_dataset(opt.data_dir)

    print(f"Number of images: {len(file_paths)}")

    transform=transforms.Compose(
                [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
    dataset = DefectDataset(file_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # R1 regularization
    if opt.R1_regularization == "True":
        print("Using R1 regularization")
        r1_regularizer = R1(weight=opt.R1_lambda)
        if cuda:
            r1_regularizer.cuda()

    # ----------
    #  Training
    # ----------
    print("Starting training...")
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            #print(f"Shape of real_imgs: {real_imgs.shape}")  # Debug

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            #print(f"Shape of generated images: {gen_imgs.shape}")  # Debug

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Add R1 penalty if enabled
            if opt.R1_regularization == "True":
                real_imgs.requires_grad = True  # Add this line
                r1_loss = r1_regularizer(discriminator(real_imgs), real_imgs)
                d_loss += r1_loss

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            """
            if batches_done % opt.sample_interval == 0:
                # Save generated images in images_dir
                save_image(gen_imgs.data[:25], os.path.join(images_dir, f"{batches_done}.png"), nrow=5, normalize=True)
            """
                
        # Save the models every 50 epochs
        #if epoch % 200 == 0 or epoch == opt.n_epochs - 1:
        if epoch == opt.n_epochs - 1:
            # Save the models
            torch.save(generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch}.pth"))
            #torch.save(discriminator.state_dict(), os.path.join(output_dir, f"discriminator_epoch_{epoch}.pth"))
            print(f"Models saved for epoch {epoch}")

if __name__ == "__main__":
    main()