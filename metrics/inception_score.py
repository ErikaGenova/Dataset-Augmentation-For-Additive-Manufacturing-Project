import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import argparse

# Dataset wrapper to ignore labels and return only images
class IgnoreLabelDataset(torch.utils.data.Dataset):
  """Dataset wrapper to ignore labels and return only images."""
  def __init__(self, orig):
      self.orig = orig

  def __getitem__(self, index):
      return self.orig[index][0]  # Return only the image, ignoring the label

  def __len__(self):
      return len(self.orig)

"""Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
"""
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):

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
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()  # Set the model to evaluation mode

    # Define an upsampling layer to resize images to 299x299 if needed
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    # Function to get predictions from the Inception model
    def get_pred(x):
        # Resize the input if required
        if resize:
            x = up(x)
        # Pass the input through the Inception model and return softmax predictions
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Initialize an array to store predictions
    preds = np.zeros((N, 1000))

    # Iterate through the DataLoader to get predictions in batches
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)  # Convert batch to the correct data type
        batchv = Variable(batch)  # Wrap batch in a Variable
        batch_size_i = batch.size()[0]  # Get the actual batch size

        # Store predictions for the current batch
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Compute the mean KL divergence for each split
    split_scores = []

    # Iterate through the number of splits
    for k in range(splits):
        # Divide predictions into splits
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)  # Compute the marginal probability p(y)
        scores = []

        # Compute KL divergence for each image in the split
        for i in range(part.shape[0]):
            pyx = part[i, :]  # Get p(y|x) for each image
            scores.append(entropy(pyx, py))  # Compute KL divergence
        split_scores.append(np.exp(np.mean(scores)))  # Compute exponential of mean KL divergence

    # Return the mean and standard deviation of the scores
    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Inception Score')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for computation', default=False)
    parser.add_argument('--dir_images', type=str, help='Directory of images', required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Inception model')
    parser.add_argument('--splits', type=int, default=10, help='Number of splits for Inception Score')
    args = parser.parse_args()

    # Load the CIFAR-10 dataset with transformations
    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),  # Resize images to 32x32
                                 transforms.ToTensor(),  # Convert images to tensors
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
                             ])
    )

    print ("Calculating Inception Score...")
    # Compute and print the Inception Score for the dataset
    print (inception_score(IgnoreLabelDataset(cifar), cuda=args.cuda, batch_size=args.batch_size, resize=True, splits=args.splits))