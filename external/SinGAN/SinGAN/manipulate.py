from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

"""
    SinGAN_generate function generates images using the trained SinGAN model.
    Prameters:
        - Gs: List of generator networks for each scale
        - Zs: List of noise tensors for each scale
        - reals: List of real images for each scale
        - NoiseAmp: List of noise amplitudes for each scale
        - opt: Options containing various parameters for the generation process
        - in_s: Input image to be used for generation (default is None, which means a zero tensor will be used)
        - scale_v: Vertical scaling factor (default is 1)
        - scale_h: Horizontal scaling factor (default is 1)
        - n: Current scale index (default is 0)
        - gen_start_scale: Starting scale for generation (default is 0)
        - num_samples: Number of samples to generate at each scale (default is 50)
    Returns:
        - I_curr: Generated image at the last scale
    
"""
def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=50):
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    
    images_cur = []

    # Iterate through the scales to generate progressively larger images
    # Each scale has its own generator (G), optimazed noise tensor (Z_opt), and noise amplitude (noise_amp)
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):

        # Calculate the padding size based on the kernel size and number of layers
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        # Calculate the size of the noise for the current scale
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        # Save the generated images from the previous scale
        images_prev = images_cur
        # Initialize the list to store the generated images for the current scale
        images_cur = []
        
        for i in range(0,num_samples,1):

            # if it is the first scale, generate a random noise tensor
            if n == 0: 
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,opt.nc_z,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                # Generate a random noise tensor for the current scale
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            # if it is the first scale, use the input image as the previous image
            if images_prev == []:
                I_prev = m(in_s)
            else:
                # Resize the previous image to match the current scale
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)

                if opt.mode != "SR":
                    # Resize the previous image to match the current scale
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if n == len(reals)-1:
                if opt.mode == 'train':
                    if opt.dir_model is not None:
                        dir2save = 'SinGANRandomSamples/%s/%s/%s' % (opt.class_, opt.input_name[:-4], opt.dir_model)
                    else:
                        dir2save = 'SinGANRandomSamples/%s/%s/scale_factor=%f,alpha=%d' % ( opt.class_, opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    # print that it is generating a random sample
                    print('Generating random sample %d/%d for %s' % (i, num_samples, opt.input_name[:-4]))
                    # add the name of image
                    plt.imsave('%s/%s_%d.png' % (dir2save, opt.input_name[:-4], i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1, cmap='gray')
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach()

