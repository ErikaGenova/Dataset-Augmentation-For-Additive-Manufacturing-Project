from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', required=True)
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    # num of channels of input image
    opt.nc_im = 1
    # num of channels of noise grayscale image
    opt.nc_z = 1
    # complete with the other missing options
    opt = functions.post_config(opt)

    # Inizialize empty structures
    Gs = [] # list of generators
    Zs = [] # list of noise images
    reals = [] # list of real images
    NoiseAmp = []   # list of noise amplitudes

    # Prepare the output's folder
    dir2save = functions.generate_dir2save(opt) # create the path to save the model

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        # load the real image from the input directory
        real = functions.read_image(opt)

        print('real shape before adjust_scales2image:', real.shape)
        # Prepare the dimensions and scale parameters to train SinGAN on my image
        functions.adjust_scales2image(real, opt)
        print('real shape after adjust_scales2image:', real.shape)
        
        # Train the model
        train(opt, Gs, Zs, reals, NoiseAmp)

        # Generate the final images
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
