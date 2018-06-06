# --*-- coding:utf-8 --*--
"""image impainting use naive dcgan"""

import sys
from boto.auth import sha256
sys.path.append(".")
from models.naive_dcgan import Generator, Discriminator, init_weights
from utils.image_utils import get_image_from_path
from __future__ import print_function
import os
import argparse
import torch
import torch.utils as tutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as tvutils
import numpy as np

import random

# define argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=64,
                    help="your input batch size")

parser.add_argument("--image_size", type=int, default=64,
                    help="height and width of your input image to D-network")

parser.add_argument("--z_dim", type=int, default=100,
                    help="dimention of latent variable of Generator Network")

parser.add_argument("--num_gf", type=int, default=64,
                    help="number of generator network filter factor")

parser.add_argument("--num_df", type=int, default=64,
                    help="number of discriminator network filter factor")

parser.add_argument("--num_iters", type=int, default=1000,
                    help="number of iterations form image impainting optimization")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate for adjusting gradient of z, default is 0.01")

parser.add_argument("--cuda", action="store_true",
                    help="enable cuda accelerating")

parser.add_argument("--num_gpus", type=int, default=1,
                    help="number of gpus to use")

parser.add_argument("--output_dir", default=".",
                    help="directory path to output intermediate images and impainted images")

parser.add_argument("--netG_path", required=True,
                    help="file path to G network,to generate the patches ")

parser.add_argument("--netD_path", required=True,
                    help="file path to D network,to impainting for more real image")

parser.add_argument("--random_seed", type=int,
                    help="you can specify the random seed if you want")

parser.add_argument("--aligned_images", type=str, narg="+",
                    help="input source aligned images for mask and impainting")

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
    print("Info: you have a cuda device available,you may specify --cuda to enable it.")

# device setting
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

num_gpu = int(args.num_gpus)

num_gf = int(args.num_gf)

num_df = int(args.num_df)

num_iters = int(args.num_iters)

z_dim = int(args.z_dim)

# number of image channels
num_channels = 3

# set cudnn benchmark=True ,for the best of the backend to find best hardware algorithms
cudnn.benchmark = True

# define Generator and Discriminator Network.

if args.cuda:
    use_cuda = True
else:
    use_cuda = False

generator = Generator(z_dim, num_gf, num_channels, use_cuda, num_gpu)
discriminator = Discriminator(num_channels, num_df, use_cuda, num_gpu)

# move generator and disciminator to cuda device if cuda is activated
generator.to(device)
discriminator.to(device)

# init weights of  g and d

generator.apply(init_weights)
discriminator.apply(init_weights)

# load net state if exists
print("load trained state dict from local files...")
generator.load_state_dict(torch.load(args.netG_path))
discriminator.load_state_dict(torch.load(args.netD_path))
print("generator and discriminator state dict loaded, done.")

print("Generator Info:")
print(generator)
print("Discriminator Info:")
print(discriminator)

# 初始化潜变量z，这个z的目的是匹配待修补的一个batch的图像在generator的底层分布来源
z = torch.randn(args.batch_size, z_dim, 1, 1).to(device)

image_shape = [args.image_size, args.image_size, num_channels]


def impainting(n_epochs):
    # 创建输出文件夹
    output_dir = args.output_dir
    source_imagedir = os.path.join(output_dir, "source_images")
    masked_imagedir = os.path.join(output_dir, "masked_images")
    impainted_imagedir = os.path.join(output_dir, "impainted_images")
    os.makedirs(source_imagedir, exist_ok=True)
    os.makedirs(masked_imagedir)
    os.makedirs(impainted_imagedir)

    # 总共需要修复多少图片
    num_images = len(args.aligned_images)
    # 总共可以分为多少的batch来进行处理
    num_batches = int(np.ceil(num_images / args.batch_size))

    for idx in range(num_batches):
        # 对于每一个batch的图片进行如下处理
        lidx = idx * args.batch_size
        hidx = min(num_images, (idx + 1) * args.batch_size)
        realBatchSize = hidx - lidx

        batch_images = [get_image_from_path(imgpath) for imgpath in args.aligned_images[lidx:hidx]]
        batch_images = np.array(batch_images).astype(np.float32)
        if realBatchSize < args.batch_size:
            print("number of batch images : ", realBatchSize)
            # 如果需要修补的图片没有一个batch那么多，用0来填充
            batch_images = np.pad(batch_images, [(0, args.batch_size - realBatchSize), (0, 0), (0, 0), (0, 0)], "constant")
            batch_images = batch_images.astype(np.float32)
        
        # 输入的原始图片已经准备好，开始准备mask
        # 暂时只提供中心mask
        mask = np.ones(shape=image_shape)
        imageCenterScale = 0.25
        lm = args.image_size * imageCenterScale
        hm = args.image_size * (1 - imageCenterScale)
        # 将图像中心mask为0
        mask[lm:hm, lm:hm, :] = 0.0
        masked_batch_images = np.multiply(batch_images, mask)

        real_rows = np.ceil(realBatchSize / 8)
        real_cols = min(8, realBatchSize)

    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            # first update discriminator's parameters
            # max(log(D(x))+log(1-D(G(z))))
            # ------------------------------
            # first train for real images

            # zero grad
            discriminator.zero_grad()
            real_batch_data = data[0].to(device)
            batch_size = real_batch_data.size(0)
            batch_labels = torch.full((batch_size,), label_real, device=device)
            # forward to get output of D for real images
            real_batch_output = discriminator(real_batch_data)
            # compute loss for D_real
            loss_D_real = criteria(real_batch_output, batch_labels)
            # backward the gradients
            loss_D_real.backward()

            loss_D_real_mean = loss_D_real.mean().item()

            # now we train with fake images generated by G
            # first sample from latent variable z
            fake_batch_z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            # then generate fake images
            fake_batch_images = generator(fake_batch_z)
            # generate fake labels
            batch_labels.fill_(label_fake)
            # forward D to get output
            # !这里使用detach,是为了防止判别网络通过生成网络的输出值拷贝改变其值,进而影响生成网络的梯度计算
            fake_batch_output = discriminator(fake_batch_images.detach())
            # compute loss of D for fake images
            loss_D_fake = criteria(fake_batch_output, batch_labels)
            # backward
            loss_D_fake.backward()

            loss_D_fake_mean = loss_D_fake.mean().item()

            loss_D_total = loss_D_fake + loss_D_real

            # update G parameters
            optim_D.step()

            # now it's G's turn
            # ---------------------------------
            # let's maximize logD(G(z))
            generator.zero_grad()
            # mark label real for G
            batch_labels.fill_(label_real)
            # 我们需要判别网络更新之后的判别结果,就像银行知道了假币的缺陷,那么制造假币的需要知道银行掌握的信息(判据)
            batch_output_G = discriminator(fake_batch_images)
            # compute loss for G
            loss_G = criteria(batch_output_G, batch_labels)
            # compute gradient for G
            loss_G.backward(retain_graph=True)

            loss_G_mean = loss_G.mean().item()

            # update parameters for G
            optim_G.step()

            print("epoch:{}/{},n_batch:{}/{},loss_real_D:{:.4},loss_fake_D:{:.4},loss_G:{:.4}".format(
                epoch, n_epochs, i, len(dataloader), loss_D_real_mean, loss_D_fake_mean, loss_G_mean))

            if i % 100 == 0:
                # sample every 100 batch
                tvutils.save_image(
                    real_batch_data, "%s/sample_real.png" % args.output_dir, normalize=True)
                fake_batch_images = generator(sample_batch_z)
                tvutils.save_image(fake_batch_images.detach(), "%s/sample_fake_images_epoch%03d_%s.png" % 
                                   (args.output_dir, epoch, args.dataset), normalize=True)


if __name__ == "__main__":
    train(num_epochs)
