# Image-impainting-with-dcgan-pytorch
project on image impatining based on context error with pytorch.with exploration of dcgans.

# TODO:

[✓] implement naive dcgan with pytorch.

[ ] fully test dcgan-pytorch

[ ] modify dcgan-pytorch to image impainting mission.

[ ] fully test image-impainting-dcgan-with-pytorch.

[ ] tranform naive dcgan-pytorch to w-dcgan-pytorch

[ ] fully test w-dcgan-pytorch

[ ] transform image-impainting-with-dcgan to image-impainting-with-wgan

[ ] fully test image-impainting-wgan

# requirements
* python3
* packages : see requirements.txt

# Usage
## First install the required packages
## Usage of naive_dcgan_main.py
This follows the original deep convolutional gan implementation.  you can use 
```
python3 naive_dcgan_main.py 
```
And then it will show help message to guide you how to use it.like:
```
usage: naive_dcgan_main.py [-h] --dataset DATASET --dataset_root DATASET_ROOT
                           [--num_workers NUM_WORKERS]
                           [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                           [--z_dim Z_DIM] [--num_gf NUM_GF] [--num_df NUM_DF]
                           [--num_epochs NUM_EPOCHS] [--lr LR] [--beta1 BETA1]
                           [--cuda] [--num_gpus NUM_GPUS]
                           [--output_dir OUTPUT_DIR] [--netG_path NETG_PATH]
                           [--netD_path NETD_PATH] [--random_seed RANDOM_SEED]
naive_dcgan_main.py: error: the following arguments are required: --dataset, --dataset_root
```
**Please be noticed that if you want to use celeba ,imagenet or your custom dataset,make sure it follows the ImageFolder role of pytorch.That is: put your real image directory under some root directory and pass the root directory to the dataset_root argument. For example:~/dataset/celebA/celebA/*.png**

### some results for naive_dcgan_main
#### 25epoch results on cifar10:  
![cifar10 25 epoch samples](https://raw.githubusercontent.com/lotuswhl/Image-inpainting-with-dcgan-pytorch/master/images/sample_fake_images/sample_fake_images_epoch024-cifar10.png)  
#### 20 epoch for celeba


