import os
import re
import sys
import glob
import torch
import argparse
from PIL import Image
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from torchvision.models import Inception_V3_Weights


"""
  get_all_file_paths(data_dir) is a function that loads file paths for all images in the dataset
""" 
def get_all_file_paths(args):
    '''Load file paths for all images in the dataset.'''

    # for original dataset
    if args.model == "original":
        classes = ['NoDefects', 'Defects']
        file_paths = []
        for idx, cls in enumerate(classes):
            folder = os.path.join(args.dir_images, cls)
            for ext in ('png', 'jpg', 'jpeg'):
                files = glob.glob(os.path.join(folder, f'*.{ext}'))
                file_paths += files
        return file_paths
    # for generated images by CVAE and GANs
    else: 
        file_paths= []

        for ext in ('png', 'jpg', 'jpeg'):
            files = glob.glob(os.path.join(args.dir_images, f'*.{ext}'))
            for path in files:
                file_paths.append(path)
        return file_paths

# Define the image transformation pipeline
image_trasforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.Resize((299, 299)),  # Resize to 299x299 for Inception v3
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Dataset wrapper to ignore labels and return only images
class IgnoreLabelDefectDataset(torch.utils.data.Dataset):
  '''Custom dataset reading files and returning only images.'''
  def __init__(self, file_paths, transform=None):
      self.file_paths = file_paths
      self.transform = transform

  def __len__(self):
      return len(self.file_paths)

  def __getitem__(self, idx):
      img_path = self.file_paths[idx]
      image = Image.open(img_path).convert('L')  # grayscale
      if self.transform:
          image = self.transform(image)
      return image


"""Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (1xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
"""
def inception_score(imgs, cuda=True, batch_size=32, splits=1):

    # Number of images
    N = len(imgs)

    # Ensure batch size and number of images are valid
    assert batch_size > 0
    assert N > batch_size

    # Set up the data type for GPU or CPU
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Create a DataLoader for batching the images
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load the pretrained Inception v3 model
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True).type(dtype)
    inception_model.eval()  # Set the model to evaluation mode

    # Initialize an array to store predictions
    preds = np.zeros((N, 1000))

    # Iterate through the DataLoader to get predictions in batches
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)  # Convert batch to the correct data type
        batchv = Variable(batch)  # Wrap batch in a Variable
        batch_size_i = batch.size()[0]  # Get the actual batch size

        # Output of the Inception model 
        x = inception_model(batchv)

        # Store predictions for the current batch
        preds[i*batch_size:i*batch_size + batch_size_i] = F.softmax(x, dim=1).data.cpu().numpy()

    # Compute the mean KL divergence for each split
    split_scores = []

    # Iterate through the number of splits
    for k in range(splits):
        # Divide predictions into splits
        part = preds[k * (N // splits): (k+1) * (N // splits), :]

        # Compute the marginal probability p(y). If it is concentrated in many classes -> the images are diversity
        py = np.mean(part, axis=0)  
        
        scores = []

        # Compute KL divergence for each image in the split
        for i in range(part.shape[0]):
            # Get p(y|x) for each image. If p(y|x) is concentrated in a few classes -> the images are realistics
            pyx = part[i, :]  

            # Compute KL divergence
            scores.append(entropy(pyx, py))  

        split_scores.append(np.exp(np.mean(scores)))  # Compute exponential of mean KL divergence

    # Return the mean and standard deviation of the scores
    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Inception Score')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for computation', default=False)
    parser.add_argument("--base_dir", type=str, help="Base directory of the projects", default="/content/mla_project/")
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--experiment", type=str, help="Experiment", default="")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Inception model')
    parser.add_argument('--splits', type=int, default=10, help='Number of splits for Inception Score')
    args = parser.parse_args()

    # control if the model is correct
    if args.model not in ("original", "VAE", "CVAE", "GANs", "Diffusion", "SinGAN"):
        print("Invalid model. Choose either 'original', 'VAE', 'CVAE', 'GANs', 'Diffusion' or 'SinGAN'")
        sys.exit()

    if args.model == "original":
        args.dir_images = os.path.join(args.base_dir, "images", "original")
    else:
        # path of the generated data based on specific model and number of experiment
        args.dir_images = os.path.join(args.base_dir, "images", "augmented", args.model, args.experiment)

    if not os.path.exists(args.dir_images):
        print(f"Directory {args.dir_images} doesn't exist. Please try again")
        sys.exit()

    # get all file paths for the dataset
    file_paths = get_all_file_paths(args)

    # Create a dataset with the file paths and transformations
    dataset = IgnoreLabelDefectDataset(file_paths, transform=image_trasforms)

    if args.model == "original":
        print (f"Calculating Inception Score on {args.model} images...")
    else:
        print (f"Calculating Inception Score on the images generated by {args.model} of {args.experiment}...")

    # Perform Inception Score calculation
    mean, std = inception_score(dataset, cuda=args.cuda, batch_size=args.batch_size, splits=args.splits)
    # Print the results
    print(f"Inception Score: {mean:.4f} ± {std:.4f}")
  