
import torch.nn as nn
import torch.nn.parallel as parallel

def init_weights(nn_model):
    class_name = nn_model.__class__.__name__
    # convolutional layer weight initialization
    if class_name.find("Conv") != -1:
        nn_model.weight.data.normal_(0.0, 0.04)
    # bacthnorm layer weight initilization
    elif class_name.find("BatchNrom") != -1:
        nn_model.weight.data.normal_(1.0, 0.04)
        nn_model.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Definition of the Generator Network
    --------------
    Parameters:
    z_dim: dimention of latent variable z for generator network
    num_gf: fiter numbers factor of genrator network
    num_channels: number channels of output image
    use_cuda: whether use cuda or not
    num_gpus: number of gpus to use
    """

    def __init__(self, z_dim, num_gf, num_channels, use_cuda=False, num_gpus=1):
        super(Generator, self).__init__()
        self.use_cuda = use_cuda
        self.num_gpus = num_gpus

        self.net = nn.Sequential(
            # input_channels,output_channels,kernel_size,stride,padding
            # input batch of z_dim vector to transpose convolution
            nn.ConvTranspose2d(z_dim, 8 * num_gf, 4, 1, 0, bias=False),
            # batch norm over feature maps
            nn.BatchNorm2d(8 * num_gf),
            nn.ReLU(True),
            # now,we got 8*num_gf x 4 x 4
            # upsample
            nn.ConvTranspose2d(8 * num_gf, 4 * num_gf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * num_gf),
            nn.ReLU(True),
            # so far,we got 4*num_gf x 8 x 8
            nn.ConvTranspose2d(4 * num_gf, 2 * num_gf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * num_gf),
            nn.ReLU(True),
            # 2*num_gf x 16 x 16
            nn.ConvTranspose2d(2 * num_gf, num_gf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gf),
            nn.ReLU(True),
            # num_gf x 32 x 32
            nn.ConvTranspose2d(num_gf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # num_channel x 64 x64
        )

    def forward(self, X):
        if self.use_cuda and self.num_gpus > 1 :
            output = parallel.data_parallel(self.net, X, range(self.num_gpus
            ))
        else:
            output = self.net(X)
        return output


class Discriminator(nn.Module):
    """
    Discriminator Network definition
    ----------------------------
    Parameters:
    num_channels: number channels of input images
    num_df: number filters factor
    use_cuda:
    num_gpus:
    """

    def __init__(self, num_channels, num_df, use_cuda=False, num_gpus=1):
        super(Discriminator,self).__init__()
        self.use_cuda = use_cuda
        self.num_gpus = num_gpus

        self.net = nn.Sequential(
            # input image is num_channels x 64 x 64
            nn.Conv2d(num_channels, num_df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # num_df x 32 x 32
            nn.Conv2d(num_df, 2 * num_df, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * num_df),
            nn.LeakyReLU(0.2, inplace=True),

            # 2*num_df x 16 x 16
            nn.Conv2d(2 * num_df, 4 * num_df, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * num_df),
            nn.LeakyReLU(0.2, inplace=True),

            # 4*num_df x 8 x 8
            nn.Conv2d(4 * num_df, 8 * num_df, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * num_df),
            nn.LeakyReLU(0.2, inplace=True),

            # 8*num_df x 4 x 4
            nn.Conv2d(8 * num_df, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()

        )

    def forward(self, X):
        if self.use_cuda and self.num_gpus > 1:
            output = parallel.data_parallel(self.net, X, range(self.num_gpus))
        else:
            output = self.net(X)
        return output.view(-1, 1).squeeze(1)

