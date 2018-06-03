# --*-- coding:utf-8 --*--
"""modified from naive_dcgan_train to set up for testing"""
from __future__ import print_function
import os
import argparse
import torch
import torch.utils as tutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as tvutils
import sys
import random
sys.path.append(".")
from models.naive_dcgan import *

# define argument parser
parser = argparse.ArgumentParser()



parser.add_argument("--batch_size", type=int, default=64,
                    help="your input batch size")

parser.add_argument("--z_dim", type=int, default=100,
                    help="dimention of latent variable of Generator Network")

parser.add_argument("--num_gf", type=int, default=64,
                    help="number of generator network filter factor")

parser.add_argument("--cuda", action="store_true",
                    help="enable cuda accelerating")

parser.add_argument("--num_gpus", type=int, default=1,
                    help="number of gpus to use")

parser.add_argument("--output_dir", default=".",
                    help="directory path to output sample images")

parser.add_argument("--netG_path", required=True,
                    help="file path to G network to restore Generator for Sampling")

parser.add_argument("--random_seed", type=int,
                    help="you can specify the random seed if you want")

args = parser.parse_args()

print("custom arguments:", args)

# make sure output dir exists
try:
    os.makedirs(args.output_dir)
except OSError:
    pass

# generate random seed if not specified
if args.random_seed is None:
    args.random_seed = random.randint(1024, 2048)

print("random seed:", args.random_seed)

random.seed(args.random_seed)

torch.manual_seed(args.random_seed)

if torch.cuda.is_available() and not args.cuda:
    print("Info: you have a cuda device available,you can sepecify --cuda to enable it.")

# device setting
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

num_gpu = int(args.num_gpus)

num_gf = int(args.num_gf)

z_dim = int(args.z_dim)

# number of image channels
num_channels = 3

# set cudnn benchmark=True ,for the best of the backend to find best hardware algorithms
cudnn.benchmark = True


if args.cuda:
    use_cuda = True
else:
    use_cuda = False

generator = Generator(z_dim, num_gf, num_channels, use_cuda, num_gpu)
# move generator and disciminator to cuda device if cuda is activated
generator.to(device)

generator.apply(init_weights)

# load net state if exists
generator.load_state_dict(torch.load(args.netG_path))

print("Generator Info:")
print(generator)


# sample a batch from generator
sample_batch_z = torch.randn(args.batch_size, z_dim, 1, 1).to(device)


def predict():
    # sample images from generator
    fake_batch_images = generator(sample_batch_z)
    tvutils.save_image(fake_batch_images.detach(), "%s/sample_from_generator_with_seed_{%d}.png" %
                                   (args.output_dir, args.random_seed), normalize=True)



if __name__ == "__main__":
    predict()
    print(
        "Sampling Done! Image saved at %s/sample_from_generator_with_seed_{%d}.png" % ((args.output_dir, args.random_seed)))
