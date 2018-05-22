# --*-- coding:utf-8 --*--
from __future__ import print_function
import torch
import torchvision
import numpy as np

import argparse

# define argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True, help="specify one of : cifar10|mnist|lsun|imagenet|custom_dir|lfw|fake")

parser.add_argument("--dataset_root", required=True, help="root dir to dataset")

parser.add_argument("--workers", type=int, default=4, help="number of workers to load data")

parser.add_argument("--batch_size", type=int, default=64, help="your input batch size")

parser.add_argument("--image_size", type=int, default=64, help="height and width of your input image to D-network")

parser.add_argument("--z_dim", type=int, default=100, help="dimention of latent variable of Generator Network")

parser.add_argument("--num_gf", type=int, default=64, help="number of generator network filter factor")

parser.add_argument("--num_df", type=int, default=64, help="number of discriminator network filter factor")

parser.add_argument("--num_epoch", type=int, default=25, help="number of training epochs of the dataset")

parser.add_argument("--lr", type=float, default=0.0002, help="learning rate , default is 0.0002")

parser.add_argument("--beta1", type=float, default=0.5, help="beta1 parameter for adam optimizer,default : 0.5")

parser.add_argument("--cuda", action="store_true", help="enable cuda accelerating")

parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus to use")

parser.add_argument("--output_dir", default=".", help="directory path to output images and checkpoint files ...")

parser.add_argument("--netG_path", default="", help="file path to G network,to continue training ")

parser.add_argument("--netD_path", default="", help="file path to D network,to continue training")

parser.add_argument("--random_seed", type=int, help="you can specify the random seed if you want")


