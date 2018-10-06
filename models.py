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
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, num_features, gated=False, gate_param=0.):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            ConvLayer(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
            ConvLayer(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.gate_param = nn.Parameter(torch.tensor(gate_param))
        self.gated = gated

    def forward(self, x):
        y = self.main(x)
        if self.gated:
            return F.sigmoid(self.gate_param)*x + y
        else:
            return x + y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=2, gated=False, gate_param=0.):
        super(Block, self).__init__()
        main = nn.Sequential()
        main.add_module('conv_{}_{}'.format(in_channels, out_channels),
                        ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
        for _ in range(num_residuals):
            main.add_module('residual_{}_{}'.format(out_channels, out_channels),
                            ResidualBlock(out_channels, gated=gated, gate_param=gate_param))
        main.add_module('bn_{}'.format(out_channels), nn.BatchNorm2d(out_channels))
        main.add_module('relu_{}'.format(out_channels), nn.ReLU(True))

        self.main = main

    def forward(self, x):
        return self.main(x)

class Unet(nn.Module):
    def __init__(self, num_features, num_residuals=2, gated=False, gate_param=0., tanh_mode=False):
        super(Unet, self).__init__()

        self.block01 = Block(1, num_features*1, num_residuals=2, gated=gated, gate_param=gate_param)
        # 101 -> 50
        self.down01 = nn.Conv2d(num_features*1, num_features*1, kernel_size=3, stride=2, padding=0, bias=False)

        self.block12 = Block(num_features*1, num_features*2, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 50 -> 25
        self.down12 = nn.Conv2d(num_features*2, num_features*2, kernel_size=2, stride=2, padding=0, bias=False)

        self.block23 = Block(num_features*2, num_features*4, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 25 -> 12
        self.down23 = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, stride=2, padding=0, bias=False)

        self.block34 = Block(num_features*4, num_features*8, num_residuals=num_residuals, gated=gated, gate_param=gate_param)
        # 12 -> 6
        self.down34 = nn.Conv2d(num_features*8, num_features*8, kernel_size=2, stride=2, padding=0, bias=False)

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

        if tanh_mode:
            self.final_acti = nn.Tanh()
        else:
            self.final_acti = nn.Sigmoid()

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
        y = self.final_acti(self.final_conv(u10))
        return y

class DCGAN_Dv0(nn.Module):
    def __init__(self, num_features, nc=2, dropout=0):
        super(DCGAN_Dv0, self).__init__()

        main = nn.Sequential(
            # 101 -> 49
            nn.Conv2d(nc, num_features*1, 5, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 49 -> 23
            nn.Conv2d(num_features*1, num_features*2, 5, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # 23 -> 10
            nn.Conv2d(num_features*2, num_features*4, 5, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 10 -> 4
            nn.Conv2d(num_features*4, num_features*8, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # 4 -> 1
            nn.Conv2d(num_features*8, 1, 4, 1, 0, bias=False)
        )

        self.main = main

    def forward(self, x):
        return self.main(x).view(-1,1)

# class DCGAN_Dv1(nn.Module):
#     def __init__(self, num_features, dropout=0):
#         super(DCGAN_Dv1, self).__init__()

#         main = nn.Sequential(
#             # input contains 2 channels (image and mask)
#             # 101 -> 49
#             nn.Conv2d(1, num_features*1, 5, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 49 -> 23
#             nn.Conv2d(num_features*1, num_features*2, 5, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             # 23 -> 10
#             nn.Conv2d(num_features*2, num_features*4, 5, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 10 -> 4
#             nn.Conv2d(num_features*4, num_features*8, 4, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             # 4 -> 1
#             nn.Conv2d(num_features*8, 1, 4, 1, 0, bias=False)
#         )

#         self.main = main

#     def forward(self, x):
#         x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
#         x = (x1 * x2)
#         return self.main(x).view(-1,1)

# class DCGAN_Dv2(nn.Module):
#     def __init__(self, num_features, dropout=0):
#         super(DCGAN_Dv2, self).__init__()

#         main = nn.Sequential(
#             # input contains 2 channels (image and mask)
#             # 101 -> 49
#             nn.Conv2d(2, num_features*1, 5, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 49 -> 23
#             nn.Conv2d(num_features*1, num_features*2, 5, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             # 23 -> 10
#             nn.Conv2d(num_features*2, num_features*4, 5, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 10 -> 4
#             nn.Conv2d(num_features*4, num_features*8, 4, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             # 4 -> 1
#             nn.Conv2d(num_features*8, 1, 4, 1, 0, bias=False)
#         )

#         self.main = main

#     def forward(self, x):
#         x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
#         x = torch.cat((x1, (x1*x2)), dim=1)
#         return self.main(x).view(-1,1)

class DCGAN_Dv1(DCGAN_Dv0):
    def __init__(self, num_features, nc=1, dropout=0):
        DCGAN_Dv0.__init__(self, num_features, nc=1, dropout=0)

    def forward(self, x):
        x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
        x = (x1 * x2)
        return self.main(x).view(-1,1)

class DCGAN_Dv2(DCGAN_Dv0):
    def __init__(self, num_features, nc=2, dropout=0):
        DCGAN_Dv0.__init__(self, num_features, nc=2, dropout=0)

    def forward(self, x):
        x1, x2 = x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1)
        x = torch.cat((x1, (x1*x2)), dim=1)
        return self.main(x).view(-1,1)


choiceD = {'v0':DCGAN_Dv0, 'v1':DCGAN_Dv1, 'v2':DCGAN_Dv2}
