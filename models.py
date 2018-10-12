import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, idim=2, odim=2, hidden_dim=512):
        super(FC, self).__init__()

        main = nn.Sequential(
            nn.Linear(idim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, odim),
        )
        self.main = main

    def forward(self, x):
        output = self.main(x)
        return output

class ConvLayer(nn.Module):
    """ Conv2d with reflection padding """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=bias)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **interpolate_kws):
        super(UpsampleConvLayer, self).__init__()
        self.interpolate_kws = interpolate_kws
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=bias)

    def forward(self, x):
        x = F.interpolate(x, **self.interpolate_kws)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

class DownsampleConvLayer(UpsampleConvLayer):
    def forward(self, x):
        # swap order of operations wrt UpsampleConvLayer
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        x = F.interpolate(x, **self.interpolate_kws)
        return x

class InterpolateLayer(nn.Module):
    def __init__(self, **kwargs):
        super(InterpolateLayer, self).__init__()
        self.kwargs = kwargs

    def forward(self, x):
        x = F.interpolate(x, **self.kwargs)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, num_features, gated=False, gate_param=0., batchnorm=True):
        super(ResidualBlock, self).__init__()

        # self.main = nn.Sequential(
        #     nn.BatchNorm2d(num_features),
        #     nn.ReLU(True),
        #     ConvLayer(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(num_features),
        #     nn.ReLU(True),
        #     ConvLayer(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        # )

        main = nn.Sequential()

        if batchnorm:
            main.add_module('bn1_{}'.format(num_features), nn.BatchNorm2d(num_features))

        main.add_module('relu1_{}'.format(num_features), nn.ReLU(True))
        main.add_module('conv1_{}_{}'.format(num_features, num_features),
                        ConvLayer(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True))

        if batchnorm:
            main.add_module('bn2_{}'.format(num_features), nn.BatchNorm2d(num_features))

        main.add_module('relu2_{}'.format(num_features), nn.ReLU(True))
        main.add_module('conv2_{}_{}'.format(num_features, num_features),
                        ConvLayer(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True))

        self.gate_param = nn.Parameter(torch.tensor(gate_param))
        self.gated = gated

    def forward(self, x):
        y = self.main(x)
        if self.gated:
            return F.sigmoid(self.gate_param)*x + y
        else:
            return x + y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=2, gated=False, gate_param=0., batchnorm=True):
        super(Block, self).__init__()
        main = nn.Sequential()

        main.add_module('conv_{}_{}'.format(in_channels, out_channels),
                        ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))

        for _ in range(num_residuals):
            main.add_module('residual_{}_{}'.format(out_channels, out_channels),
                            ResidualBlock(out_channels, gated=gated, gate_param=gate_param, batchnorm=batchnorm))

        if batchnorm:
            main.add_module('bn_{}'.format(out_channels), nn.BatchNorm2d(out_channels))

        main.add_module('relu_{}'.format(out_channels), nn.ReLU(True))

        self.main = main

    def forward(self, x):
        return self.main(x)

class Unetv0(nn.Module):
    def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0., sigmoid=True):
        super(Unetv0, self).__init__()

        self.block01 = Block(1, num_features*1, num_residuals=2, gated=gated, gate_param=gate_param)
        # 101 -> 50
        self.down01 = nn.MaxPool2d(2)

        self.block12 = Block(num_features*1, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 50 -> 25
        self.down12 = nn.MaxPool2d(2)

        self.block23 = Block(num_features*2, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 25 -> 12
        self.down23 = nn.MaxPool2d(2)

        self.block34 = Block(num_features*4, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 12 -> 6
        self.down34 = nn.MaxPool2d(2)

        # middle
        self.block44 = Block(num_features*8, num_features*16, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 6 -> 12
        self.up43 = nn.ConvTranspose2d(num_features*16, num_features*8, kernel_size=2, stride=2, padding=0, bias=False)
        self.block43 = Block(num_features*16, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 12 -> 25
        self.up32 = nn.ConvTranspose2d(num_features*8, num_features*4, kernel_size=3, stride=2, padding=0, bias=False)
        self.block32 = Block(num_features*8, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 25 -> 50
        self.up21 = nn.ConvTranspose2d(num_features*4, num_features*2, kernel_size=2, stride=2, padding=0, bias=False)
        self.block21 = Block(num_features*4, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 50 -> 101
        self.up10 = nn.ConvTranspose2d(num_features*2, num_features*1, kernel_size=3, stride=2, padding=0, bias=False)
        self.block10 = Block(num_features*2, num_features*1, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        self.final_conv = nn.Conv2d(num_features*1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.final_acti = nn.Sigmoid()

        self.sigmoid = sigmoid

    def forward(self, x):
        b01 = self.block01(x)
        d01 = self.down01(b01)  # 101 -> 50
        b12 = self.block12(d01)
        d12 = self.down12(b12)  # 50 -> 25
        b23 = self.block23(d12)
        d23 = self.down23(b23)  # 25 -> 12
        b34 = self.block34(d23)
        d34 = self.down34(b34)  # 12 -> 6
        b44 = self.block44(d34)  # middle
        u43 = self.up43(b44)  # 6 -> 12
        u43 = torch.cat([u43,b34], dim=1)
        b43 = self.block43(u43)
        u32 = self.up32(b43)  # 12 -> 25
        u32 = torch.cat([u32,b23], dim=1)
        b32 = self.block32(u32)
        u21 = self.up21(b32)  # 25 -> 50
        u21 = torch.cat([u21,b12], dim=1)
        b21 = self.block21(u21)
        u10 = self.up10(b21)  # 50 -> 101
        u10 = torch.cat([u10,b01], dim=1)
        b10 = self.block10(u10)
        # y = self.final_acti(self.final_conv(b10))
        y = self.final_conv(b10)
        if self.sigmoid:
            y = self.final_acti(y)
        return y

class Unetv0a(Unetv0):
    def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0., sigmoid=True):
        Unetv0.__init__(self, num_features, num_residuals, gated, gate_param, sigmoid)
        self.down01 = nn.Conv2d(num_features*1, num_features*1, kernel_size=3, stride=2, padding=0, bias=False)
        self.down12 = nn.Conv2d(num_features*2, num_features*2, kernel_size=2, stride=2, padding=0, bias=False)
        self.down23 = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, stride=2, padding=0, bias=False)
        self.down34 = nn.Conv2d(num_features*8, num_features*8, kernel_size=2, stride=2, padding=0, bias=False)

# class Unetv1(nn.Module):
#     def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0.):
#         super(Unetv1, self).__init__()

#         self.block01 = Block(1, num_features*1, num_residuals=2, gated=gated, gate_param=gate_param)
#         # 101 -> 99
#         self.down01 = nn.Conv2d(num_features*1, num_features*1, kernel_size=3, stride=1, padding=0, bias=False)

#         self.block12 = Block(num_features*1, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
#         # 99 -> 33
#         self.down12 = nn.MaxPool2d(3)

#         self.block23 = Block(num_features*2, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
#         # 33 -> 11
#         self.down23 = nn.MaxPool2d(3)

#         self.block34 = Block(num_features*4, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
#         # 11 -> 9
#         self.down34 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, stride=1, padding=0, bias=False)

#         # middle
#         self.block44 = Block(num_features*8, num_features*16, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

#         # 9 -> 11
#         self.up43 = nn.ConvTranspose2d(num_features*16, num_features*8, kernel_size=3, stride=1, padding=0, bias=False)
#         self.block43 = Block(num_features*16, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

#         # 11 -> 33
#         self.up32 = UpsampleConvLayer(num_features*8, num_features*4, scale_factor=3)
#         self.block32 = Block(num_features*8, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

#         # 33 -> 99
#         self.up21 = UpsampleConvLayer(num_features*4, num_features*2, scale_factor=3)
#         self.block21 = Block(num_features*4, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

#         # 99 -> 101
#         self.up10 = nn.ConvTranspose2d(num_features*2, num_features*1, kernel_size=3, stride=1, padding=0, bias=False)
#         self.block10 = Block(num_features*2, num_features*1, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

#         self.final_conv = nn.Conv2d(num_features*1, 1, kernel_size=1, stride=1, padding=0, bias=True)
#         self.final_acti = nn.Sigmoid()

#     def forward(self, x):
#         b01 = self.block01(x)
#         d01 = self.down01(b01)  # 101 -> 99
#         b12 = self.block12(d01)
#         d12 = self.down12(b12)  # 99 -> 33
#         b23 = self.block23(d12)
#         d23 = self.down23(b23)  # 33 -> 11
#         b34 = self.block34(d23)
#         d34 = self.down34(b34)  # 11 -> 9
#         b44 = self.block44(d34)  # middle
#         u43 = self.up43(b44)  # 9 -> 11
#         u43 = torch.cat([u43,b34], dim=1)
#         b43 = self.block43(u43)
#         u32 = self.up32(b43)  # 11 -> 33
#         u32 = torch.cat([u32,b23], dim=1)
#         b32 = self.block32(u32)
#         u21 = self.up21(b32)  # 33 -> 99
#         u21 = torch.cat([u21,b12], dim=1)
#         b21 = self.block21(u21)
#         u10 = self.up10(b21)  # 99 -> 101
#         y = self.final_acti(self.final_conv(u10))
#         return y

class Unetv1(Unetv0):
    def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0., sigmoid=True):
        Unetv0.__init__(self, num_features, num_residuals, gated, gate_param, sigmoid)

        # 101 -> 99
        self.down01 = nn.Conv2d(num_features*1, num_features*1, kernel_size=3, stride=1, padding=0, bias=False)
        # 99 -> 33
        self.down12 = nn.MaxPool2d(3)
        # 33 -> 11
        self.down23 = nn.MaxPool2d(3)
        # 11 -> 9
        self.down34 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, stride=1, padding=0, bias=False)
        # 9 -> 11
        self.up43 = nn.ConvTranspose2d(num_features*16, num_features*8, kernel_size=3, stride=1, padding=0, bias=False)
        # 11 -> 33
        self.up32 = UpsampleConvLayer(num_features*8, num_features*4, bias=False, scale_factor=3)
        # 33 -> 99
        self.up21 = UpsampleConvLayer(num_features*4, num_features*2, bias=False, scale_factor=3)
        # 99 -> 101
        self.up10 = nn.ConvTranspose2d(num_features*2, num_features*1, kernel_size=3, stride=1, padding=0, bias=False)

class Unetv1a(Unetv1):
    def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0., sigmoid=True):
        Unetv1.__init__(self, num_features, num_residuals, gated, gate_param, sigmoid)

        self.down12 = nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=3, padding=0, bias=False)
        self.down23 = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, stride=3, padding=0, bias=False)

choiceF = {'v0':Unetv0, 'v0a':Unetv0a, 'v1':Unetv1, 'v1a':Unetv1a}

class DCGAN_Dv0(nn.Module):
    def __init__(self, num_features, nc=2, dropout=0):
        super(DCGAN_Dv0, self).__init__()

        main = nn.Sequential(
            # 101 -> 99
            nn.Conv2d(nc, num_features*1, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 99 -> 33
            DownsampleConvLayer(num_features*1, num_features*2, bias=False, scale_factor=1./3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # 33 -> 11
            DownsampleConvLayer(num_features*2, num_features*4, bias=False, scale_factor=1./3),
            nn.LeakyReLU(0.2, inplace=True),
            # 11 -> 9
            nn.Conv2d(num_features*4, num_features*8, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # 9 -> 1
            nn.Conv2d(num_features*8, 1, 9, 1, 0, bias=False)
        )
        self.main = main

    def forward(self, x):
        x = 2.0 * (x - 0.5)  # [0,1] -> [-1,1]
        return self.main(x).view(-1,1)

class DCGAN_Dv0a(DCGAN_Dv0):
    def __init__(self, num_features, nc=1, dropout=0):
        DCGAN_Dv0.__init__(self, num_features, nc, dropout)

    def forward(self, x):
        x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
        x = (x1 * x2)
        return DCGAN_Dv0.forward(self, x)

class DCGAN_Dv0b(DCGAN_Dv0):
    def __init__(self, num_features, nc=2, dropout=0):
        DCGAN_Dv0.__init__(self, num_features, nc, dropout)

    def forward(self, x):
        x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
        x = torch.cat((x1, (x1*x2)), dim=1)
        return DCGAN_Dv0.forward(self, x)

class DCGAN_Dv1(nn.Module):
    """ Like v0 but conv instead of downsampleconv """
    def __init__(self, num_features, nc=2, dropout=0):
        super(DCGAN_Dv1, self).__init__()

        main = nn.Sequential(
            # 101 -> 99
            nn.Conv2d(nc, num_features*1, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 99 -> 33
            nn.Conv2d(num_features*1, num_features*2, 3, 3, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # 33 -> 11
            nn.Conv2d(num_features*2, num_features*4, 3, 3, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 11 -> 9
            nn.Conv2d(num_features*4, num_features*8, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # 9 -> 1
            nn.Conv2d(num_features*8, 1, 9, 1, 0, bias=False)
        )
        self.main = main

    def forward(self, x):
        x = 2.0 * (x - 0.5)  # [0,1] -> [-1,1]
        return self.main(x).view(-1,1)

class DCGAN_Dv1a(DCGAN_Dv1):
    def __init__(self, num_features, nc=1, dropout=0):
        DCGAN_Dv1.__init__(self, num_features, nc, dropout)

    def forward(self, x):
        x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
        x = (x1 * x2)
        DCGAN_Dv1.forward(self, x)

class DCGAN_Dv1b(DCGAN_Dv1):
    def __init__(self, num_features, nc=2, dropout=0):
        DCGAN_Dv1.__init__(self, num_features, nc, dropout)

    def forward(self, x):
        x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
        x = torch.cat((x1, (x1*x2)), dim=1)
        DCGAN_Dv1.forward(self, x)

class DCGAN_Gv0(nn.Module):
    def __init__(self, num_features, nz, nc=1):
        super(DCGAN_Gv0, self).__init__()

        main = nn.Sequential(
            # 1 -> 9
            nn.ConvTranspose2d(nz, num_features*8, 9, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(True),
            # 9 -> 11
            nn.ConvTranspose2d(num_features*8, num_features*4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(True),
            # 11 -> 33
            UpsampleConvLayer(num_features*4, num_features*2, bias=False, scale_factor=3),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(True),
            # 33 -> 99
            UpsampleConvLayer(num_features*2, num_features*1, bias=False, scale_factor=3),
            nn.BatchNorm2d(num_features*1),
            nn.ReLU(True),
            # 99 -> 101
            nn.ConvTranspose2d(num_features*1, nc, 3, 1, 0, bias=False),
            nn.Tanh()
        )
        self.main = main

    def forward(self, z):
        x = self.main(z)
        x = 0.5 * (x + 1.0)  # [-1,1] -> [0,1]
        return x

class DCGAN_Gv0a(nn.Module):
    def __init__(self, num_features, nz, nc=1):
        super(DCGAN_Gv0a, self).__init__()

        main = nn.Sequential(
            # 1 -> 9
            nn.ConvTranspose2d(nz, num_features*8, 9, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(True),
            # 9 -> 11
            nn.ConvTranspose2d(num_features*8, num_features*4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(True),
            # 11 -> 33
            nn.ConvTranspose2d(num_features*4, num_features*2, 3, 3, 0, bias=False),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(True),
            # 33 -> 99
            nn.ConvTranspose2d(num_features*2, num_features*1, 3, 3, 0, bias=False),
            nn.BatchNorm2d(num_features*1),
            nn.ReLU(True),
            # 99 -> 101
            nn.ConvTranspose2d(num_features*1, nc, 3, 1, 0, bias=False),
            nn.Tanh()
        )
        self.main = main

    def forward(self, z):
        x = self.main(z)
        x = 0.5 * (x + 1.0)  # [-1,1] -> [0,1]
        return x

class DCGAN_Gv1(nn.Module):
    def __init__(self, num_features, nz, nc=1, output_size=101):
        super(DCGAN_Gv1, self).__init__()

        main = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose2d(nz, num_features*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*16),
            nn.ReLU(True),
            # 4 -> 8
            UpsampleConvLayer(num_features*16, num_features*8, bias=False, scale_factor=2),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(True),
            # 8 -> 16
            UpsampleConvLayer(num_features*8, num_features*4, bias=False, scale_factor=2),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(True),
            # 16 -> 32
            UpsampleConvLayer(num_features*4, num_features*2, bias=False, scale_factor=2),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(True),
            # 32 -> 64
            UpsampleConvLayer(num_features*2, num_features*1, bias=False, scale_factor=2),
            nn.BatchNorm2d(num_features*1),
            nn.ReLU(True),
            # 64 -> 128
            UpsampleConvLayer(num_features*1, nc, bias=False, scale_factor=2),
            nn.Tanh(),
            InterpolateLayer(size=output_size)
        )
        self.main = main

    def forward(self, z):
        x = self.main(z)
        x = 0.5 * (x + 1.0)  # [-1,1] -> [0,1]
        return x

class DCGAN_Dv2(nn.Module):
    def __init__(self, num_features, nc, input_size=128, dropout=0):
        super(DCGAN_Dv2, self).__init__()

        main = nn.Sequential(
            InterpolateLayer(size=input_size),
            # 128 -> 64
            nn.Conv2d(nc, num_features*1, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 64 -> 32
            nn.Conv2d(num_features*1, num_features*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            # 32 -> 16
            nn.Conv2d(num_features*2, num_features*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 16 -> 8
            nn.Conv2d(num_features*4, num_features*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            # 8 -> 4
            nn.Conv2d(num_features*8, num_features*16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 4 -> 1
            nn.Conv2d(num_features*16, 1, 4, 1, 0, bias=False)
        )
        self.main = main

    def forward(self, x):
        x = 2.0 * (x - 0.5)  # [0,1] -> [-1,1]
        return self.main(x)

choiceG = {'v0':DCGAN_Gv0, 'v0a':DCGAN_Gv0a, 'v1':DCGAN_Gv1}
choiceD = {'v0':DCGAN_Dv0, 'v0a':DCGAN_Dv0a, 'v0b':DCGAN_Dv0b,
           'v1':DCGAN_Dv1, 'v1a':DCGAN_Dv1a, 'v1b':DCGAN_Dv1b,
           'v2':DCGAN_Dv2}

class UnetPhi(nn.Module):
    def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0., dropout=0):
        super(UnetPhi, self).__init__()

        self.block01 = Block(1, num_features*1, num_residuals=2, gated=gated, gate_param=gate_param, batchnorm=False)
        # 101 -> 99
        self.down01 = nn.Conv2d(num_features*1, num_features*1, kernel_size=3, stride=1, padding=0, bias=False)
        self.dropout01 = nn.Dropout(dropout)

        self.block12 = Block(num_features*1, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param, batchnorm=False)
        # 99 -> 33
        self.down12 = nn.MaxPool2d(3)
        self.dropout12 = nn.Dropout(dropout)

        self.block23 = Block(num_features*2, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 33 -> 11
        self.down23 = nn.MaxPool2d(3)
        self.dropout23 = nn.Dropout(dropout)

        self.block34 = Block(num_features*4, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 11 -> 9
        self.down34 = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, stride=1, padding=0, bias=False)
        self.dropout34 = nn.Dropout(dropout)

        # middle
        self.block44 = Block(num_features*8, num_features*16, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 9 -> 11
        self.up43 = nn.ConvTranspose2d(num_features*16, num_features*8, kernel_size=3, stride=1, padding=0, bias=False)
        self.dropout43 = nn.Dropout(dropout)
        self.block43 = Block(num_features*16, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 11 -> 33
        self.up32 = UpsampleConvLayer(num_features*8, num_features*4, scale_factor=3)
        self.dropout32 = nn.Dropout(dropout)
        self.block32 = Block(num_features*8, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 33 -> 99
        self.up21 = UpsampleConvLayer(num_features*4, num_features*2, scale_factor=3)
        self.dropout21 = nn.Dropout(dropout)
        self.block21 = Block(num_features*4, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

        # 99 -> 101
        self.up10 = nn.ConvTranspose2d(num_features*2, num_features*1, kernel_size=3, stride=1, padding=0, bias=False)
        self.dropout10 = nn.Dropout(dropout)
        self.block10 = Block(num_features*2, num_features*1, num_residuals=num_residuals, gated=gated, gate_param=gate_param)

    def forward(self, x):
        b01 = self.block01(x)
        d01 = self.down01(b01)  # 101 -> 99
        d01 = self.dropout01(d01)
        b12 = self.block12(d01)
        d12 = self.down12(b12)  # 99 -> 33
        d12 = self.dropout12(d12)
        b23 = self.block23(d12)
        d23 = self.down23(b23)  # 33 -> 11
        d23 = self.dropout23(d23)
        b34 = self.block34(d23)
        d34 = self.down34(b34)  # 11 -> 9
        d34 = self.dropout34(d34)
        b44 = self.block44(d34)  # middle
        u43 = self.up43(b44)  # 9 -> 11
        u43 = torch.cat([u43,b34], dim=1)
        u43 = self.dropout43(u43)
        b43 = self.block43(u43)
        u32 = self.up32(b43)  # 11 -> 33
        u32 = torch.cat([u32,b23], dim=1)
        u32 = self.dropout32(u32)
        b32 = self.block32(u32)
        u21 = self.up21(b32)  # 33 -> 99
        u21 = torch.cat([u21,b12], dim=1)
        u21 = self.dropout21(u21)
        b21 = self.block21(u21)
        u10 = self.up10(b21)  # 99 -> 101
        u10 = torch.cat([u10,b01], dim=1)
        u10 = self.dropout10(u10)
        b10 = self.block10(u10)
        return b10