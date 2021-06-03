import torch
import torch.nn as nn
import models.basicblock as B
# 还是不好的话，可以去掉那两次注意力，把nc改成128
class D_Block(nn.Module):
    def __init__(self, channel_in, channel_out, deconv = False):
        super(D_Block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.PReLU()
        # if deconv: # down层采用deconv，up层采用conv
        #     self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.),
        #                             kernel_size=3, stride=1, padding=2, bias=True, dilation=2)
        # 还是都卷积吧
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1,
                                padding=1)
        self.relu3 = nn.PReLU()
        self.tail = B.conv(channel_in, channel_out, mode='CBR')
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        out = self.tail(out)
        return out


# change pool to conv
class _down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(_down, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=4, stride=2, padding=1)

        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))

        return out

# PixelShuffle + nolocal + 1x1 conv
class _up(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True, upFactor=2):
        super().__init__()
        assert in_channels%4 == 0
        self.up = nn.PixelShuffle(upscale_factor=upFactor)
        self.conv2 =B.conv(in_channels=int(in_channels/4), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias,mode='CR')

    def forward(self,x):
        out = self.up(x)
        out = self.conv2(out)
        return out

class Net(nn.Module):
    def __init__(self, in_nc=1, out_nc=3, nc=64, nb=17, act_mode='BR'):
        super(Net, self).__init__()
        self.head = B.conv(in_nc, nc, 1, 1, 0, mode='C')
        self.up = _up(nc,nc//4)
        self.down = _down(nc,nc*4)
        # main block
        self.up_layer1_1 = D_Block(nc//4, nc//4)
        self.up_layer1_2 = D_Block(nc//4, nc//4)
        self.down_layer1_1 = D_Block(nc, nc, True)
        self.down_layer1_2 = D_Block(nc, nc, True)

        # main block
        self.up1 = _up(nc,nc//4)
        self.down1 = _down(nc//4,nc)
        self.att_up1 = B.eca_layer(nc//2)
        self.att_down1 = B.eca_layer(nc*2)
        self.up_layer2_1 = D_Block(nc//2, nc//2)
        self.up_layer2_2 = D_Block(nc//2, nc//2)
        self.down_layer2_1 = D_Block(nc*2, nc*2, True)
        self.down_layer2_2 = D_Block(nc*2, nc*2, True)
        # self.down_layer2_3 = D_Block(nc*2, nc*2, True)

        # main block
        self.up2 = _up(nc*2,nc//2)
        self.down2 = _down(nc//2,nc*2)
        self.att_up2 = B.eca_layer(nc)
        self.att_down2 = B.eca_layer(nc*4)
        self.up_layer3_1 = D_Block(nc, nc)
        self.up_layer3_2 = D_Block(nc, nc)
        self.down_layer3_1 = D_Block(nc*4, nc*4, True)
        self.down_layer3_2 = D_Block(nc*4, nc*4, True)

        self.att = B.eca_layer(nc*8)
        tail = [
                B.conv(nc*8, nc*2, mode='CBR'),
                B.conv(nc*2, nc, mode='CBR'),
                D_Block(nc, nc, True),
                B.conv(nc, in_nc, 1, 1, 0, mode='C')
                ]
        self.tail = B.sequential(*tail)

    def forward(self, x): # 非residual
        # b,c,h,w = x.shape
        # x = x.expand(b,512,h,w)
        # 彩图和真实图像用卷积
        residual = x

        x = self.head(x)
        x_up = self.up(x)

        x_up = self.up_layer1_1(x_up)
        x_up = self.up_layer1_2(x_up)

        x_down = self.down_layer1_1(x)
        x_down = self.down_layer1_2(x_down)

        # 第一次交叉
        cross_up1 = self.up1(x_down)
        cross_down1 = self.down1(x_up)
        cat_up1 = torch.cat([x_up, cross_up1],1)
        cat_up1 = self.att_up1(cat_up1)

        cat_down1 = torch.cat([x_down,cross_down1],1)
        cat_down1 = self.att_down1(cat_down1)

        x_up1 = self.up_layer2_1(cat_up1)
        x_up1 = self.up_layer2_2(x_up1)

        x_down1 = self.down_layer2_1(cat_down1)
        x_down1 = self.down_layer2_2(x_down1)
        # x_down1 = self.down_layer2_3(x_down1)

        # 第二次交叉
        cross_up2 = self.up2(x_down1)
        cross_down2 = self.down2(x_up1)
        cat_up2 = torch.cat([x_up1, cross_up2],1)
        cat_up2 = self.att_up2(cat_up2)
        cat_down2 = torch.cat([x_down1,cross_down2],1)
        cat_down2 = self.att_down2(cat_down2)

        x_up2 = self.up_layer3_1(cat_up2)
        x_up2 = self.up_layer3_2(x_up2)

        x_down2 = self.down_layer3_1(cat_down2)
        x_down2 = self.down_layer3_2(x_down2)

        # 最后一层
        last = self.down(x_up2)
        last = torch.cat([last, x_down2],1)
        last = self.att(last)
        out = self.tail(last)
        out = torch.add(residual, out)

        return out
