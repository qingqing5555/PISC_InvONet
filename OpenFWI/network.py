import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from DeformConv import DeformConv2d
from HWD import Down_wt
from Conv2Former import ConvMod

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


def load_dict_to_model(pre_dict, model):
    """
    Transfer Learning
:param pre_FilePath: Pre-trained model
New Model Path
    :return:
    """
    # model = torch.load(model_path)
    model_dict = model.state_dict()  # Get model_to model parameters
    # Filter
    new_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    print('Total : {}, update: {}'.format(len(pre_dict), len(new_dict)))
    model.load_state_dict(model_dict)
    print("loaded finished!")
    return model


class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea,
                 kernel_size=3, stride=1, padding=1,
                 bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)


class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)


class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        # if out_fea > 1:
        #     layers.append(ScConv(out_fea))  
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeformConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(DeformConvBlock, self).__init__()
        layers = [DeformConv2d(inc=in_fea, outc=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# -------------------------------------Residual Module--------------------------------------
#Normal Residual
class ResidualBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', use_1x1conv=False,
                 relu_slop=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=kernel_size,
                               padding=padding)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels=in_fea, out_channels=out_fea, kernel_size=1, stride=stride) 
        else:
            self.conv3 = None

        self.BN1 = NORM_LAYERS[norm](out_fea)
        self.BN2 = NORM_LAYERS[norm](out_fea)
        self.LeakyReLU = nn.LeakyReLU(relu_slop, inplace=True)

    def forward(self, x):

        y = self.LeakyReLU(self.BN1(self.conv1(x))) 
        y = self.BN2(self.conv2(y)) 
        if self.conv3:
            x = self.conv3(x)  
        y = self.LeakyReLU(y + x)
        return y

# 可变性卷积残差
class Def_ResidualBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', use_1x1conv=False,
                 relu_slop=0.2):
        super(Def_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        # self.conv2 = nn.Conv2d(in_channels=out_fea, out_channels=out_fea, kernel_size=kernel_size,
        #                        padding=padding)
        self.conv2 = DeformConv2d(out_fea, out_fea, kernel_size=3)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels=in_fea, out_channels=out_fea, kernel_size=1, stride=stride)  #Residual connection
        else:
            self.conv3 = None

        self.BN1 = NORM_LAYERS[norm](out_fea)
        self.LeakyReLU = nn.LeakyReLU(relu_slop, inplace=True)

    def forward(self, x):

        y = self.LeakyReLU(self.BN1(self.conv1(x))) 
        y = self.conv2(y) 
        if self.conv3:
            x = self.conv3(x)   
        y = self.LeakyReLU(y + x)
        return y

# 自注意力残差
class Mod_ResidualBlock(nn.Module):
    def __init__(self, in_fea, out_fea, stride=1, padding_D=0, padding_Skip=0, norm='bn', use_1x1conv=False,
                 relu_slop=0.2):
        super(Mod_ResidualBlock, self).__init__()
        self.conv1_1 = Down_wt(in_fea, out_fea, padding=padding_D)
        self.conv1_2 = ConvBlock(out_fea, out_fea)
        self.conv2 = ConvMod(out_fea)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels=in_fea, out_channels=out_fea, kernel_size=1, stride=stride, padding=padding_Skip)  
        else:
            self.conv3 = None

        self.BN1 = NORM_LAYERS[norm](out_fea)
        self.BN2 = NORM_LAYERS[norm](out_fea)
        self.LeakyReLU = nn.LeakyReLU(relu_slop, inplace=True)

    def forward(self, x):
        y = self.conv1_1(x) 
        y = self.conv1_2(y)
        y = self.conv2(y)  
        if self.conv3:
            x = self.conv3(x)   
        y = self.LeakyReLU(y + x)
        return y
# ----------------------------------------Network Design----------------------------------------

class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))

        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(2, 0))
        self.convblock5_2 = ConvBlock(dim3, dim3)

        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)

        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        # self.convblock7_1 = DeformConv2d(dim4, dim4, stride=2)
        # self.convblock7_2 = DeformConv2d(dim4, dim4)

        # self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=5, padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)

        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)

        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)

        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)

        x = self.convblock7_1(x)  # (None, 256, 8, 9)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)

        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part 
        # x = self.deconv1_1(x)  # (None, 512, 5, 5)
        # x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x

# ------------------------Add Down_wt ConvMod
class ConvModNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ConvModNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))

        self.wtconvblock5_1 = Down_wt(dim3, dim3, 2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.ConvMod5_3 = ConvMod(dim3)

        self.wtconvblock6_1 = Down_wt(dim3, dim4, 2)
        self.Dconvblock6_2 = DeformConv2d(dim4, dim4, kernel_size=3)
        self.ConvMod6_3 = ConvMod(dim4)

        self.wtconvblock7_1 = Down_wt(dim4, dim4, 2)
        self.Dconvblock7_2 = DeformConv2d(dim4, dim4, kernel_size=3)
        self.ConvMod7_3 = ConvMod(dim4)

        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)

        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)

        x = self.wtconvblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.ConvMod5_3(x)

        x = self.wtconvblock6_1(x)  # (None, 256, 16, 18)
        x = self.Dconvblock6_2(x)  # (None, 256, 16, 18)
        x = self.ConvMod6_3(x)

        x = self.wtconvblock7_1(x)  # (None, 256, 8, 9)
        x = self.Dconvblock7_2(x)  # (None, 256, 8, 9)
        x = self.ConvMod7_3(x)

        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x

# 动态卷积 + 自注意力
class Df_ConvModNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Df_ConvModNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = DeformConv2d(dim2, dim2, kernel_size=3, padding=(1, 0))

        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = DeformConv2d(dim2, dim2, kernel_size=3, padding=(1, 0))

        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = DeformConv2d(dim3, dim3, kernel_size=3, padding=(1, 0))

        self.wtconvblock5_1 = Down_wt(dim3, dim3, 2)
        self.ConvMod5_2 = ConvMod(dim3)

        self.wtconvblock6_1 = Down_wt(dim3, dim4, 2)
        self.ConvMod6_2 = ConvMod(dim4)

        self.wtconvblock7_1 = Down_wt(dim4, dim4, 2)
        self.ConvMod7_2 = ConvMod(dim4)

        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)

        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)

        x = self.wtconvblock5_1(x)  # (None, 128, 32, 35)
        x = self.ConvMod5_2(x)

        x = self.wtconvblock6_1(x)  # (None, 256, 16, 18)
        x = self.ConvMod6_2(x)

        x = self.wtconvblock7_1(x)  # (None, 256, 8, 9)
        x = self.ConvMod7_2(x)

        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x

#Self-attention
class baseline_mod(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(baseline_mod, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.ConvMod1_1 = ConvMod(dim1)

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        # self.ConvMod2_3 = ConvMod(dim2)

        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        # self.ConvMod3_3 = ConvMod(dim2)

        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 3), padding=(1, 1))
        # self.ConvMod4_3 = ConvMod(dim3)

        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(2, 0))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.ConvMod5_3 = ConvMod(dim3)

        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.ConvMod6_3 = ConvMod(dim4)

        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.ConvMod7_3 = ConvMod(dim4)

        # self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=5, padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.ConvMod1_1(x)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)

        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)

        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.ConvMod5_3(x)

        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        x = self.ConvMod6_3(x)

        x = self.convblock7_1(x)  # (None, 256, 8, 9)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        x = self.ConvMod7_3(x)

        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part
        # x = self.deconv1_1(x)  # (None, 512, 5, 5)
        # x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x

# Dynamic Convolution
class baseline_df(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(baseline_df, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        # self.convblock1_2 = ConvBlock(dim1, dim1, kernel_size=(3, 3), padding=(1, 1))

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))

        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        self.convblock3_2 = DeformConv2d(dim2, dim2, kernel_size=3)

        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 3), padding=(1, 1))
        self.convblock4_2 = DeformConv2d(dim3, dim3, kernel_size=3)

        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(2, 0))
        self.convblock5_2 = ConvBlock(dim3, dim3)

        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)

        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)

        # self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=5, padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)

        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)

        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)

        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)

        x = self.convblock7_1(x)  # (None, 256, 8, 9)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)

        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Decoder Part
        # x = self.deconv1_1(x)  # (None, 512, 5, 5)
        # x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x


class baseline_df_mod(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(baseline_df_mod, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.ConvMod1_1 = ConvMod(dim1)

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))

        # self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock3_2 = DeformConv2d(dim2, dim2, kernel_size=3)
        self.convblock3 = Def_ResidualBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
                                            use_1x1conv=True)

        # self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        # self.convblock4_2 = DeformConv2d(dim3, dim3, kernel_size=3)
        self.convblock4 = Def_ResidualBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
                                            use_1x1conv=True)

        self.convblock5 = Mod_ResidualBlock(dim3, dim3, stride=2, padding_D=(1, 0), padding_Skip=(2, 0), use_1x1conv=True)

        self.convblock6 = Mod_ResidualBlock(dim3, dim4, stride=2, use_1x1conv=True)

        self.convblock7 = Mod_ResidualBlock(dim4, dim4, stride=2, use_1x1conv=True)
        self.convblock7_1 = Down_wt(dim4, dim4)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.ConvMod7_3 = ConvMod(dim4)

        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=5, padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.ConvMod1_1(x)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3(x)  # (None, 64, 125, 70)

        x = self.convblock4(x)  # (None, 64, 63, 70)

        x = self.convblock5(x)  # (None, 128, 34, 35)

        x = self.convblock6(x)  # (None, 128, 34, 35)

        x = self.convblock7(x)  # (None, 256, 9, 9)

        x = self.convblock8(x)  # (None, 512, 5, 5)

        # Decoder Part
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x

class ConvModNet_SEG(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ConvModNet_SEG, self).__init__()
        self.convblock1_1 = ConvBlock(30, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))

        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.wtconvblock4_1 = Down_wt(dim2, dim3, 2)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))

        self.wtconvblock5_1 = Down_wt(dim3, dim3, 2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.ConvMod5_3 = ConvMod(dim3)

        self.wtconvblock6_1 = Down_wt(dim3, dim4, 2)
        self.Dconvblock6_2 = DeformConv2d(dim4, dim4, kernel_size=3)
        self.ConvMod6_3 = ConvMod(dim4)

        self.wtconvblock7_1 = Down_wt(dim4, dim4, 2)
        self.Dconvblock7_2 = DeformConv2d(dim4, dim4, kernel_size=3)
        self.ConvMod7_3 = ConvMod(dim4)

        self.convblock8_0 = ConvBlock(dim4, dim5, kernel_size=(6, 4), padding=1)  # 这里改了

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5 = ConvBlock(dim2, dim1)  # 这里改了
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Encoder Part  16 30 1000 150
        x = self.convblock1_1(x)  # (None, 32, 500, 150)

        x = self.convblock2_1(x)  # (None, 64, 250, 150)
        x = self.convblock2_2(x)  # (None, 64, 250, 150)

        x = self.convblock3_1(x)  # (None, 64, 125, 150)
        x = self.convblock3_2(x)  # (None, 64, 125, 150)

        x = self.convblock4_1(x)  # (None, 128, 63, 150)
        x = self.convblock4_2(x)  # (None, 128, 63, 150)

        x = self.wtconvblock5_1(x)  # (None, 128, 32, 75)
        x = self.convblock5_2(x)  # (None, 128, 32, 75)
        x = self.ConvMod5_3(x)

        x = self.wtconvblock6_1(x)  # (None, 256, 16, 38)
        x = self.ConvMod6_3(x)

        x = self.wtconvblock7_1(x)  # (None, 256, 8, 19)
        # x = self.Dconvblock7_2(x)  # (None, 256, 8, 19)
        x = self.ConvMod7_3(x)

        x = self.convblock8_0(x)  # (None, 512, 5, 18)

        # Decoder Part
        x = self.deconv1_1(x)  # (None, 512, 13,39)
        x = self.deconv1_2(x)  # (None, 512, 13,39)
        x = self.deconv2_1(x)  # (None, 256, 26, 78)
        x = self.deconv2_2(x)  # (None, 256, 26, 78)
        x = self.deconv3_1(x)  # (None, 128, 52, 126)
        x = self.deconv3_2(x)  # (None, 128, 52, 126)
        x = self.deconv4_1(x)  # (None, 64, 104, 312)
        x = self.deconv4_2(x)  # (None, 64, 104, 312)
        x = self.deconv5(x)  # (None, 32, 104, 312)
        x = F.pad(x, [-6, -6, -2, -2], mode="constant", value=0)  # (None, 32, 70, 70)   100,300
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x

class df_mod_ONET(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(df_mod_ONET, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.ConvMod1_1 = ConvMod(dim1)

        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))

        self.convblock3 = Def_ResidualBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
                                            use_1x1conv=True)

        self.convblock4 = Def_ResidualBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
                                            use_1x1conv=True)

        self.convblock5 = Mod_ResidualBlock(dim3, dim3, stride=2, padding_D=(1, 0), padding_Skip=(2, 0), use_1x1conv=True)

        self.convblock6 = Mod_ResidualBlock(dim3, dim4, stride=2, use_1x1conv=True)

        self.convblock7 = Mod_ResidualBlock(dim4, dim4, stride=2, use_1x1conv=True)
        self.convblock7_1 = Down_wt(dim4, dim4)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.ConvMod7_3 = ConvMod(dim4)

        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=5, padding=0)

        self.layer1 = nn.Sequential(nn.Linear(5, 50), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(50, 50), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(50, 50), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(50, 16), nn.ReLU())

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x, y):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.ConvMod1_1(x)

        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)

        x = self.convblock3(x)  # (None, 64, 125, 70)

        x = self.convblock4(x)  # (None, 64, 63, 70)

        x = self.convblock5(x)  # (None, 128, 34, 35)

        x = self.convblock6(x)  # (None, 128, 34, 35)

        x = self.convblock7(x)  # (None, 256, 9, 9)

        x = self.convblock8(x)  # (None, 512, 5, 5)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)  # 16 16
        # test_x = torch.rand([16, 512 * 5 * 5])
        # test_y = torch.rand([16, 16])
        x = torch.matmul(y, x.reshape(16, 512 * 5 * 5))  # 16, 512*5*5
        x = x.reshape(16, 512, 5, 5)

        # Decoder Part
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x


class DConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(DConvBlock, self).__init__()
        layers = [torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding))]  # 加了频谱归一化
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = DConvBlock(1, dim1, stride=2)
        self.convblock1_2 = DConvBlock(dim1, dim1)
        self.convblock2_1 = DConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = DConvBlock(dim2, dim2)
        self.convblock3_1 = DConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = DConvBlock(dim3, dim3)
        self.convblock4_1 = DConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = DConvBlock(dim4, dim4)
        self.convblock5 = DConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


# -------------------------------------Model Type--------------------------------------
model_dict = {
    'InversionNet': InversionNet, 
    'Discriminator': Discriminator,
    'baseline_mod': baseline_mod, 
    'baseline_df': baseline_df, 
    'baseline_df_mod': baseline_df_mod, 
    'ConvModNet_SEG': ConvModNet_SEG, 
    'ConvModNet': ConvModNet,
    'df_mod_ONET': df_mod_ONET  
}

def plotimg(input1, vmin, vmax, input2=None, SaveFigPath=None):
    """
    plot img
    """
    img1 = input1
    img2 = input2
    if input2:
        fig1, ax1 = plt.subplots(2)
        fig1.set_figheight(6)
        fig1.set_figwidth(12)
        plt.subplot(1, 2, 1)
        plt.imshow(img1.cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                   cmap='jet')
        plt.colorbar()
        plt.title('img1')
        plt.subplot(1, 2, 2)
        plt.imshow(img2, vmin=vmin, vmax=vmax,
                   cmap='jet')
        plt.colorbar()
        plt.title('img2')
    else:
        fig1, ax1 = plt.subplots(1)
        fig1.set_figheight(6)
        fig1.set_figwidth(12)
        plt.imshow(img1.numpy(), vmin=vmin, vmax=vmax,
                   cmap='jet')
        plt.colorbar()
        plt.title('img')
    if SaveFigPath:
        plt.savefig(SaveFigPath + 'init_model.png')
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    input = torch.rand([16, 5, 1000, 70])
    set = torch.rand([16, 5])
    model = df_mod_ONET()
    output = model(input, set)
    print(output.shape)
    print(count_parameters(model))
