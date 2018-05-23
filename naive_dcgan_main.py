# --*-- coding:utf-8 --*--
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
import sys
import random
sys.path.append(".")
from models.naive_dcgan import *

# define argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True,
                    help="specify one of : cifar10|mnist|celeba|lsun|imagenet|custom_dataset|lfw|fake")

parser.add_argument("--dataset_root", required=True,
                    help="root dir to dataset")

parser.add_argument("--num_workers", type=int, default=4,
                    help="number of workers to load data")

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

parser.add_argument("--num_epochs", type=int, default=25,
                    help="number of training epochs of the dataset")

parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate , default is 0.0002")

parser.add_argument("--beta1", type=float, default=0.5,
                    help="beta1 parameter for adam optimizer,default : 0.5")

parser.add_argument("--cuda", action="store_true",
                    help="enable cuda accelerating")

parser.add_argument("--num_gpus", type=int, default=1,
                    help="number of gpus to use")

parser.add_argument("--output_dir", default=".",
                    help="directory path to output images and checkpoint files ...")

parser.add_argument("--netG_path", default="",
                    help="file path to G network,to continue training ")

parser.add_argument("--netD_path", default="",
                    help="file path to D network,to continue training")

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
    print("WANGING: you have a cuda device available,you may specify --cuda to enable it.")

# prepare datasets
if args.dataset in ["imagenet", "lfw", "custom_dataset","celeba"]:
    dataset = datasets.ImageFolder(args.dataset_root, transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
# below ,we can use dataset from torchvision directly
elif args.dataset == "lsun":
    dataset = datasets.LSUN(root=args.dataset_root, classes=['bedroom_train'], transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
elif args.dataset == "cifar10":
    dataset = datasets.CIFAR10(root=args.dataset_root, download=True, transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
elif args.dataset == "fake":
    dataset = datasets.FakeData(image_size=(
        3, args.image_size, args.image_size), transform=transforms.ToTensor())

# make sure dataset have been initialized properly
assert dataset

# dataset if ready,and we need dataloader
dataloader = tutils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_workers))

# device setting
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

num_gpu = int(args.num_gpus)

num_gf = int(args.num_gf)

num_df = int(args.num_df)

num_epochs = int(args.num_epochs)

num_workers = int(args.num_workers)

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
if args.netG_path != "":
    generator.load_state_dict(torch.load(args.netG_path))
if args.netD_path != "":
    discriminator.load_state_dict(torch.load(args.netD_path))

print("Generator Info:")
print(generator)
print("Discriminator Info:")
print(discriminator)

criteria = nn.BCELoss()

# every time we will use this to sample from generator, so that we can compare the performance of training
sample_batch_z = torch.randn(args.batch_size, z_dim, 1, 1).to(device)

label_real = 1
label_fake = 0

optim_G = optim.Adam(generator.parameters(), lr=args.lr,
                     betas=(args.beta1, 0.999))
optim_D = optim.Adam(discriminator.parameters(),
                     lr=args.lr, betas=(args.beta1, 0.999))


def train(n_epochs):
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
            loss_G.backward()

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
                tvutils.save_image(fake_batch_images.detach(), "%s/sample_fake_images_epoch%03d.png" %
                                   (args.output_dir, epoch), normalize=True)

        torch.save(discriminator.state_dict(),
                   "%s/discriminator_epoch_%03d.pth" % (args.output_dir, epoch))
        torch.save(generator.state_dict(),
                   "%s/generator_epoch_%03d.pth" % (args.output_dir, epoch))


if __name__ == "__main__":
    train(num_epochs)
