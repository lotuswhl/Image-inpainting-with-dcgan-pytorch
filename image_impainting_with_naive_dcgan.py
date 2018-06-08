# --*-- coding:utf-8 --*--
"""image impainting use naive dcgan"""
from __future__ import print_function
import sys
sys.path.append(".")
from models.naive_dcgan import Generator, Discriminator, init_weights
from utils.image_utils import get_tensor_image, save_tensor_images
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

parser.add_argument("--lamd", type=float,default=0.1,
                    help="lamd ,coefficients of perceptual loss")

parser.add_argument("--aligned_images", type=str, nargs="+",
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

lamd = float(args.lamd)

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

image_shape = [num_channels, args.image_size, args.image_size]

criteria = nn.BCELoss()


def impainting():
    # 创建输出文件夹
    output_dir = args.output_dir
    source_imagedir = os.path.join(output_dir, "source_images")
    masked_imagedir = os.path.join(output_dir, "masked_images")
    impainted_imagedir = os.path.join(output_dir, "impainted_images")
    os.makedirs(source_imagedir, exist_ok=True)
    os.makedirs(masked_imagedir,exist_ok=True)
    os.makedirs(impainted_imagedir,exist_ok=True)

    # 总共需要修复多少图片
    num_images = len(args.aligned_images)
    # 总共可以分为多少的batch来进行处理
    num_batches = int(np.ceil(num_images / args.batch_size))

    for idx in range(num_batches):
        # 对于每一个batch的图片进行如下处理
        lidx = idx * args.batch_size
        hidx = min(num_images, (idx + 1) * args.batch_size)
        realBatchSize = hidx - lidx

        batch_images = [get_tensor_image(imgpath) for imgpath in args.aligned_images[lidx:hidx]]
        batch_images = torch.stack(batch_images).to(device)
        # if realBatchSize < args.batch_size:
        #     print("number of batch images : ", realBatchSize)
        #     # 如果需要修补的图片没有一个batch那么多，用0来填充
        #     batch_images = np.pad(batch_images, [(0, args.batch_size - realBatchSize), (0, 0), (0, 0), (0, 0)], "constant")
        #     batch_images = batch_images.astype(np.float32)
        
        # 输入的原始图片已经准备好，开始准备mask
        # 暂时只提供中心mask
        mask = torch.ones(size=image_shape).to(device)
        imageCenterScale = 0.3
        lm = int(args.image_size * imageCenterScale)
        hm = int(args.image_size * (1 - imageCenterScale))
        # 将图像中心mask为0
        mask[:,lm:hm, lm:hm] = 0.0
        masked_batch_images = torch.mul(batch_images, mask).to(device)

        # 先保存一下原始图片和masked图片
        save_tensor_images(batch_images.detach(),
                   os.path.join(source_imagedir,"source_image_batch_{}.png".format(idx)))
    
        save_tensor_images(masked_batch_images.detach(), os.path.join(masked_imagedir, "masked_image_batch_{}.png".format(idx)))

       
        z_hat = torch.rand(size=[realBatchSize,z_dim,1,1],dtype=torch.float32,requires_grad=True,device=device)
        z_hat.data.mul_(2.0).sub_(1.0)
        opt = optim.Adam([z_hat],lr=args.lr)       
        print("start impainting iteration for batch : {}".format(idx))
        v=torch.tensor(0,dtype=torch.float32,device=device)
        m=torch.tensor(0,dtype=torch.float32,device=device)
        
        for iteration in range(args.num_iters):
            # 对每一个batch的图像分别迭代impainting
            if z_hat.grad is not None:
                z_hat.grad.data.zero_()
            generator.zero_grad()
            discriminator.zero_grad()
            batch_images_g = generator(z_hat)
            batch_images_g_masked = torch.mul(batch_images_g,mask) 
            impainting_images = torch.mul(batch_images_g,(1-mask))+masked_batch_images
            if iteration % 100==0:
                # 保存impainting 图片结果
                print("\nsaving impainted images for batch: {} , iteration:{}".format(idx,iteration))
                save_tensor_images(impainting_images.detach(), os.path.join(impainted_imagedir,"impainted_image_batch_{}_iteration_{}.png".format(idx,iteration)))

            loss_context = torch.norm(
                (masked_batch_images-batch_images_g_masked),p=1)
            dis_output = discriminator(impainting_images)
#             print(dis_output)
            batch_labels = torch.full((realBatchSize,), 1, device=device)
            loss_perceptual = criteria(dis_output,batch_labels)
            
            total_loss = loss_context + lamd*loss_perceptual
            print("\r batch {} : iteration : {:4} , context_loss:{:.4f},percptual_loss:{:4f}".format(idx,iteration,loss_context,loss_perceptual),end="")
            total_loss.backward()
            opt.step()
#             g = z_hat.grad
#             if g is None:
#                 print("g is None")
#                 continue
#             vpre = v.clone()
#             mpre = m.clone() 
#             m = 0.99*mpre+(1-0.99)*g
#             v = 0.999*vpre+(1-0.999)*(g*g)
#             m_hat = m/(1-0.99**(iteration+1))
#             v_hat = v/(1-0.999**(iteration+1)) 
#             z_hat.data.sub_(m_hat/(torch.sqrt(v_hat)+1e-8))
#             z_hat.data = torch.clamp(z_hat.data,min=-1.0,max=1.0).to(device)



if __name__ == "__main__":
    impainting()
