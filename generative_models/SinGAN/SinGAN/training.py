import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

"""
    SinGAN is a GAN (Generative Adversarial Network) that learns to generate realistic images from a single image, 
    building a pyramid of scales from coarser (low-res) to finer (high-res). 
    The training is done progressively, one scale at a time. 
"""

"""
    The function `train` is responsible for training the SinGAN model.
"""
def train(opt, Gs, Zs, reals, NoiseAmp):
    # Read the real image and create a pyramid of images at different scales
    real_ = functions.read_image(opt)

    # Resize the image to a maximum size of 512x512 pixels
    real_ = imresize(real_, min(512 / real_.shape[2], 512 / real_.shape[3]), opt=opt)
    print('Image size: %d x %d' % (real_.shape[2], real_.shape[3]))

    # Resize the image to the specified scale1
    real = imresize(real_, opt.scale1, opt)
    # Create a pyramid of images at different scales
    reals = functions.creat_reals_pyramid(real,reals,opt)
    
    # The input image for the generator, initialized to 0 because it will be updated during training (PyTorch tensor)
    in_s = 0      
    
    scale_num = 0   # The current scale number
    nfc_prev = 0    # Number of filters in the previous scale

    print('Number of scales: %d' % (len(reals)-1))

    print('stop_scale: %d' % (opt.stop_scale))
    # Loop through the scales and train the model at each scale
    while scale_num < opt.stop_scale+1:
        print(f'Image size at scale {scale_num}: {reals[scale_num].shape[2]}x{reals[scale_num].shape[3]}' )

        # Complexity of the model is increased at each scale, higher scales have more filters and more layers
        
        # The number of filters is increased by a factor of 2 for every 4 scales
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        # Create a directory to save the model and results
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        print('out_: %s' % (opt.out_))
        print("outf: %s" % (opt.outf))
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass
        
        # Save the real image at the current scale, the real reference image to be used for training
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1, cmap='gray')

        # Initialize the generator and discriminator models at the current scale
        D_curr, G_curr = init_models(opt)
        
        # If the model has the same complexity as the previous one, load the weights from the previous scale
        if (nfc_prev==opt.nfc):
            # Every 4 scales, it enters here
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        # Train the model at the current scale
        z_curr,in_s,G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)

        # After training, the generator and discriminator are set to evaluation mode and their gradients are reset
        # Evaluation of the generator
        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()

        # Evaluation of the discriminator
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        # Accumulate the generator, noise, and noise amplitude for each scale
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        """
            Save pth of:
                - Gs: pth of the generator
                - Zs: pth of the noise
                - reals: pth of the real images
                - NoiseAmp: pth of the noise amplitude
        """
        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
        
        # Update the scale number and the previous number of filters
        scale_num+=1
        nfc_prev = opt.nfc

        # Delete the current generator and discriminator to free up memory
        del D_curr, G_curr
        torch.cuda.empty_cache()
    return


"""
    The function `train_single_scale` is responsible for training the model at a single scale.
    Parameters:
        - netD: Discriminator model
        - netG: Generator model
        - reals: Real images at different scales
        - Gs: List of generator models at different scales
        - Zs: List of noise tensors at different scales
        - in_s: Input image for the generator
        - NoiseAmp: List of noise amplitudes at different scales
        - opt: Options and hyperparameters for training
        - centers: Optional parameter for painting mode
    
    opt.nzx and opt.nzy are used to determine the size of the noise that is passed to the generator. 
    The generator creates images starting from a noise vector of defined dimensions. 
    Therefore, the size of the input to the generator (the noise) must be consistent with the dimensions of the image that the model is trying to generate. 
    If the width and height of the images are different, these values must be updated accordingly.
"""
def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    # Select the real image for the current scale
    real = reals[len(Gs)]

    # Set up the noise spatial dimensions for the generator and discriminator
    opt.nzx = real.shape[2] # Width of the image
    opt.nzy = real.shape[3] # Height of the image

    # Perform the receptive field, it depends on the kernel size, number of layers, and stride
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    
    # Set the padding size for the generator and discriminator 
    # Determine the quantity of padding needed for the noise to have the same dimensions as the image
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2) 
    # Determine the quantity of padding needed for the image to have the same dimensions as the noise
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    # Create zero padding layers for the noise and image 
    m_noise = nn.ZeroPad2d(int(pad_noise)) 
    m_image = nn.ZeroPad2d(int(pad_image))
 
    alpha = opt.alpha

    # Generate of the fixed noise, it is used to generate like base to generate synthetic images during the training
    fixed_noise = functions.generate_noise(
        [  opt.nc_z,  # number of channels in the noise
            opt.nzx,  # width of the image
            opt.nzy   # height of the image
        ],
        device=opt.device
    )

    # `z_opt` is a tensor of zeros with the same shape as the fixed noise, it will be optimized during training for the generator
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    # the m_noise padding is applied to adapt the noise size to the model size
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    
    # setup scheduler to reduce the learning rate after a certain number of epochs
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    """
        At the beginning of each epoch, noise is generated that will be used by the generator to create synthetic images.
        Where:
            - `z_opt` is the optimized noise that will be used to generate reference synthetic images during training
            - `noise_` is random noise generated at the beginning of each epoch, it is used to generate synthetic images during training
            - `m_noise` is a zero padding layer that is applied to the noise.
    """
    for epoch in range(opt.niter):
        
        # If there are no previous scales, generate noise with the same size as the image
        if (Gs == []) & (opt.mode != 'SR_train'):
            # Generate random noise with the same size as the image
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            # Expand the noise to match the number of channels in the generator
            z_opt = m_noise(z_opt.expand(1,opt.nc_z,opt.nzx,opt.nzy))
            
            # noise_ represents the random noise that will be used to generate synthetic images
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,opt.nc_z,opt.nzx,opt.nzy))
        else:
            # z_opt is not generated 
            # Generate random noise with the same size as the image
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)
            
        ###############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###############################
        for j in range(opt.Dsteps):
            """
                The discriminator is updated in two main phases: with real images and with generated images.
                The goal is to maximize the discriminator's ability to distinguish between real and fake images:
                   - Maximize the score for real images (errD_real: D(x))
                   - Minimize the score for fake images (errD_fake: D(G(z)))

                The discriminator is trained to output a high value for real images and a low value for fake images.
            """
            # ============== train with real images
            netD.zero_grad()
            
            # Enable gradient calculation respect to real images, which is necessary for the R1 penalty
            real.requires_grad_(True)

            # Discriminator evaluates real images and computes the loss
            output = netD(real).to(opt.device)

            # The real images loss is calculated as the negative mean of the discriminator output
            # This is because we want to maximize the discriminator output for real images
            errD_real = -output.mean()

            errD_real.backward(retain_graph=True)
            
            D_x = -errD_real.item()

            #================== train with fake
            # if it is the first epoch and the first step of the generator
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'): # At the first scale 
                    # `prev` is initialized to zeros with the same size as the noise
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    # save `prev` like starting point for the generator
                    in_s = prev
                    # apply the zero padding to the prev
                    prev = m_image(prev)

                    # `z_prev` is the noise that will be used to reconstruct the image, it is initialized to zeros with the same size as the noise
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    # apply the zero padding to the z_prev
                    z_prev = m_noise(z_prev)

                    # `noise_amp` regulates the amount of noise added to the image
                    opt.noise_amp = 1
                else:
                    # construct the `prev` image using previous generators, noise, reals images, noise amplitude and the input image
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    # apply the zero padding to the prev
                    prev = m_image(prev)
                    # Same as above, but with 'rec' mode, means noise optimized instead of random noise
                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                    
                    # Perform the MSE loss between the real image and the previous image
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev)) # Helps to understand how much noise to add in the new scale
                    # The noise amplitude is adjusted based on the RMSE value
                    opt.noise_amp = opt.noise_amp_init*RMSE

                    # Padding is applied to the z_prev
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_ # generator receives only pure noise
            else:
                # The noise is generated by adding the random noise and the previous image
                noise = opt.noise_amp * noise_ / noise_.std() + prev / prev.std()

            # noise is passed to the generator, which generates a fake image
            fake = netG(
                    noise.detach(), # introduces the details of the image
                    prev            # provides the structural content of the image
                )
            """ 
                `detach()` on noise: serve to prevent the Discriminator gradients from modifying the Generator at this step. 
                We only want to train D here.
            """
            
            # discriminator evaluates the fake image generated by netG and computes the loss
            # `output`` is a score: how much D "believes" that `fake` is real
            output = netD(fake.detach()) # detach() because we don't want to update the Generator at this time
            
            # We want to minimize errD_fake
            errD_fake = output.mean()
            # backpropagation 
            errD_fake.backward(retain_graph=True)

            D_G_z = output.mean().item()

            # Gradient penalty is calculated to enforce the Lipschitz constraint on the discriminator
            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device) # ensures the stability of the training
            gradient_penalty.backward()

            # Loss for the discriminator is the sum of the real and fake losses, plus the gradient penalty
            errD = errD_real+errD_fake+gradient_penalty # We have to minimize this value
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z)) 
        ###########################
 
        for j in range(opt.Gsteps):
            netG.zero_grad()

            # `output` is a score: how much D "believes" that `fake` is real
            output = netD(fake)
            
            # We want to maximize output.mean() but we minimize the loss because we use the Adam optimizer
            errG = -output.mean() 
            errG1=errG.detach().requires_grad_(True)
            errG1.backward(retain_graph=True)

            if alpha!=0:
                # instantiate another MSE loss function 
                loss = nn.MSELoss()
                # `z_opt` is combined with z_prev and scaled by noise_amp
                Z_opt = opt.noise_amp * z_opt + z_prev # it is the input of the generator
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 50 == 0 or epoch == (opt.niter-1):
            print(f'scale {len(Gs)}:[{epoch}/{opt.niter}] - errD: {errD.item():.4f}, errG: {errG.item():.4f}')

        """
            This part of the code is responsible for saving the generated images and the current state of the model at regular intervals.
            Save the generated images and the current state of the model at regular intervals (every 500 epochs or at the last epoch).
            The generated images include the fake image, the generated image from the optimized noise, and the discriminator maps for real and fake images.
            The optimized noise is saved as a .pth file for later use.

            The images saved are:
                - fake_sample.png: the generated image from the generator using the optimized noise.
                - G(z_opt).png: the generated image from the generator using the optimized noise and the previous image.
                - prev.png: the previous image used as input to the generator.
                - noise.png: the noise image used as input to the generator.
                - z_prev.png: the previous noise image used as input to the generator.
        """
        if epoch % 250 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1, cmap='gray')
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1, cmap='gray')

        if epoch == (opt.niter-1):
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG     

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, opt.nc_z, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    #print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)

    return netD, netG
