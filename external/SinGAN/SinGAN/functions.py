import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize
import os
import random
from sklearn.cluster import KMeans


# custom weights initialization called on netG and netD

"""
    The read_image(opt) function reads an image from disk and converts it into a format suitable for processing in a PyTorch mode
"""
def read_image(opt):
    # Read the image from the specified input directory and name
    x = img.imread('%s%s' % (opt.input_dir,opt.input_name))

    # If the image is grayscale (2D), add a channel dimension
    return np2torch(x, opt)

"""
    The denorm(x) function denormalizes the input tensor x to a range between 0 and 1.
"""
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


"""
    The norm(x) function normalizes the input tensor x to a range between -1 and 1.
"""
def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

"""
    The convert_image_np(inp) function converts a PyTorch tensor to a NumPy array,
    denormalizing it and adjusting the shape for visualization.
    It handles both RGB and grayscale images, ensuring the output is in the correct format for display.
"""
def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[0, 0, :, :])  # Prendi direttamente il primo (e unico) canale
        inp = inp.numpy()  # Convertilo in un array NumPy
        inp = np.transpose(inp, (0, 1))  # Lascia la forma (h, w) per grayscale

    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu, receptive_feild, ncs, epoch_num, file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

"""
    The generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1) function generates noise tensors of a specified size and type.
    It can create Gaussian noise, Gaussian mixture noise, or uniform noise.
    The function also allows for upsampling the noise to a larger size.
    The generated noise can be used for various purposes, such as initializing weights in neural networks or augmenting data.
"""
def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        # generate noise with standard deviation 1 and mean 0
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        # perform upsampling to the original size
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

"""
    This function moves a tensor to the GPU if available, otherwise it returns the tensor unchanged.
"""
def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

"""
    The np2torch(x, opt) function converts a NumPy grayscale image (x) into a PyTorch tensor, 
    normalizing it and adapting it to the format and device (CPU or GPU) required by the model.
"""
def np2torch(x, opt):
    # Se RGB, converti in grayscale
    if x.ndim == 3 and x.shape[2] == 3:
        x = color.rgb2gray(x)

    # Se è 2D (H, W), aggiungi un asse canale
    if x.ndim == 2:
        x = x[:, :, None]  # -> [H, W, 1]

    # Ora x ha shape [H, W, 1], aggiungiamo batch e channel
    x = x.transpose(2, 0, 1)  # -> [1, H, W]
    x = x[None, :, :, :]      # -> [1, 1, H, W]
    x = x / 255.0

    x = torch.from_numpy(x)
    if not opt.not_cuda:
        x = move_to_gpu(x)
        x = x.type(torch.cuda.FloatTensor)
    else:
        x = x.type(torch.FloatTensor)

    x = norm(x)

    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

"""
    The create_reals_pyramid function builds a pyramid of images (reals) starting from the original resolution image (real).
    It scales the image down by a factor of scale_factor for each level of the pyramid,
    and appends the scaled images to the reals list.
    The function returns the list of scaled images (reals).
    The function uses the imresize function to resize the image and the opt object to determine the scaling factor and other parameters.
"""
def creat_reals_pyramid(real,reals,opt):
    real = real[:, :1, :, :]
    for i in range(0, opt.stop_scale+1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals


def load_trained_pyramid(opt, mode_='train', dir=None):
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode

    print('opt.mode: %s' % opt.mode)
    if dir is None:
        dir = generate_dir2save(opt)

    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir, weights_only=False)
        Zs = torch.load('%s/Zs.pth' % dir, weights_only=False)
        reals = torch.load('%s/reals.pth' % dir, weights_only=False)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir, weights_only=False)
    else:
        print("%s does not exist" % dir)
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

"""
    The generate_dir2save(opt) function creates a directory path (as a string) where the trained models, results, 
    or outputs will be saved based on the mode specified in the opt object.
    The function uses different patterns to generate paths depending on what task or operation you're performing.
"""
def generate_dir2save(opt):
    dir2save = None

    # Training Modes
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'SinGANTrainedModels/%s/%s/scale_factor=%f,alpha=%d' % (opt.class_, opt.input_name[:-4], opt.scale_factor_init,opt.alpha)

    # Random Samples Modes
    elif opt.mode == 'random_samples':
        if opt.dir_model is not None:
            dir2save = 'SinGANRandomSamples/%s/%s/%s' % (opt.class_, opt.input_name[:-4], opt.dir_model)
        else:
            dir2save = 'SinGANRandomSamples/%s/%s/scale_factor=%f,alpha=%d' % (opt.class_, opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        if opt.dir_model is not None:
            dir2save = 'SinGANRandomSamples_ArbitrerySizes/%s/%s/%s' % (opt.class_, opt.input_name[:-4], opt.dir_model)
        else:
            dir2save = 'SinGANRandomSamples_ArbitrerySizes/%s/%s/scale_factor=%f,alpha=%d' % (opt.class_, opt.input_name[:-4], opt.scale_factor_init, opt.alpha)

    # join with the output directory opt.out
    if opt.out is not None:
        dir2save = os.path.join(opt.out, dir2save)
    return dir2save

"""
    This function prepares and adjusts some settings before starting the training of SinGAN
"""
def post_config(opt):
    # init fixed parameters

    # Set the device (CPU or GPU). If the option opt.not_cuda is True, it forces the use of the CPU.
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter # number of epochs (iterations) to train each scale of SinGAN
    opt.noise_amp_init = opt.noise_amp # amplitude (strength) of the noise added to the image during training
    opt.nfc_init = opt.nfc # number of filters in the first layer of the generator and discriminator
    opt.min_nfc_init = opt.min_nfc # minimum number of filters in the first layer of the generator and discriminator
    opt.scale_factor_init = opt.scale_factor # scale factor for the image pyramid

    # Builds a path to save the trained model
    opt.out_ = 'SinGANTrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    # seed is used to ensure the reproducibility of the results
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num


