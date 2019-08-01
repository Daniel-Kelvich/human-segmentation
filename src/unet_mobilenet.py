from copy import deepcopy
from src.mobilenet_v2 import MobileNetV2

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.nn import functional as F


class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                                stride=1, padding=0, bias=None):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, padding=padding, stride = stride,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class UNetMobileNet(nn.Module):

    def __init__(self, num_classes=1, num_filters=32, dropout_2d=0.2,
             pretrained=True, is_deconv=True, mobilenet_weights_path=""):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = MobileNetV2(n_class=1000)
        bottom_channel_nr = 160

        if pretrained:
            # Load weights into the project directory
            state_dict = torch.load(
                mobilenet_weights_path)  # add map_location='cpu' if no gpu
            self.encoder.load_state_dict(state_dict)
        self.features = deepcopy(self.encoder.features[:16])

        self.enc0 = nn.Sequential(*deepcopy(self.features[0:2]))
        self.enc1 = nn.Sequential(*deepcopy(self.features[2:4]))
        self.enc2 = nn.Sequential(*deepcopy(self.features[4:7]))
        self.enc3 = nn.Sequential(*deepcopy(self.features[7:11]))
        self.enc4 = nn.Sequential(*deepcopy(self.features[11:]))

        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 *2, num_filters * 8, False)

        self.dec5 = DecoderBlockV(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 2,   is_deconv)
        self.dec4 = DecoderBlockV(128, num_filters * 8, num_filters * 2, is_deconv)
        self.dec3 = DecoderBlockV(96, num_filters * 4, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV(88, num_filters * 2, num_filters * 2, is_deconv)
        self.dec1 = DecoderBlockV(num_filters * 2 , num_filters, num_filters * 2, is_deconv)
        self.dec0 = ConvRelu(num_filters * 10, num_filters * 2)
        self.final = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=self.dropout_2d)

    def forward(self, x):
        conv1 = self.enc0(x)
        conv2 = self.enc1(conv1)
        conv3 = self.enc2(conv2)

        conv4 = self.enc3(conv3)
        conv5 = self.enc4(conv4)

        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)

        hypercol = torch.cat((
            dec1,
            F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False)
        ), 1)

        dec0 = self.dec0(self.dropout(hypercol))

        mask = self.final(dec0)
        return mask

class DecoderBlockV(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)



class DecoderCenter(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderCenter, self).__init__()
        self.in_channels = in_channels


        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
        nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)

            )

    def forward(self, x):
        return self.block(x)