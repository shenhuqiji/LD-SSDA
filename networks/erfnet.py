# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from networks import HT

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)


    def forward(self, input):
        #output = self.conv1x1_1(input)

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        #output = self.conv1x1_2(output)

        return F.relu(output + input)

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        # self.mlp = Mlp(128)
        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(bottleneck_1d(64, 0.1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        # for x in range(0, 2):  # 2 times
        #     self.layers.append(bottleneck_1d(128, 0.1, 1))
        #     self.layers.append(bottleneck_1d(128, 0.1, 2))
        #     self.layers.append(bottleneck_1d(128, 0.1, 3))
        #     self.layers.append(bottleneck_1d(128, 0.1, 5))

        for x in range(0, 2):  # 2 times
            self.layers.append(bottleneck_1d(128, 0.1, 2))
            self.layers.append(bottleneck_1d(128, 0.1, 4))
            self.layers.append(bottleneck_1d(128, 0.1, 8))
            self.layers.append(bottleneck_1d(128, 0.1, 16))

        # only for encoder mode:
        # self.CAT_HTIHT = HT.CAT_HTIHT(vote_index,inplanes=128,outplanes=64)


        self.boundary = nn.Conv2d(128, 256, 1, stride=1, padding=0, bias=True)


    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        # output_mlp = self.mlp(output)
        # output_HT = self.CAT_HTIHT(output_mlp)
        output_boundary = self.boundary(output)

        return output, output_boundary


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        # self.conv1x1 = nn.Conv2d(ninput+noutput, noutput, (1, 1), stride=1, bias=True)
        # self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        # output = torch.cat([self.conv(input), self.upsampling(input)], 1)
        # output = self.conv1x1(output)
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(bottleneck_1d(64, 0, 1))
        self.layers.append(bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(bottleneck_1d(16, 0, 1))
        self.layers.append(bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ERFNet(nn.Module):
    def __init__(self, num_classes):  # use encoder to pass pretrained encoder
        super().__init__()

        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ERFNet, self).train(mode)

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []

        addtional_weight = []
        addtional_bias = []
        addtional_bn = []

        # print(self.modules())

        for m in self.encoder.modules():  # self.base_model.modules()
            if isinstance(m, nn.Conv2d):
                # print(1)
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                # print(2)
                base_bn.extend(list(m.parameters()))

        for m in self.decoder.modules():  # self.base_model.modules()
            if isinstance(m, nn.Conv2d):
                # print(1)
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                # print(2)
                base_bn.extend(list(m.parameters()))

        return [
            {
                'params': addtional_weight,
                'lr_mult': 10,
                'decay_mult': 1,
                'name': "addtional weight"
            },
            {
                'params': addtional_bias,
                'lr_mult': 20,
                'decay_mult': 1,
                'name': "addtional bias"
            },
            {
                'params': addtional_bn,
                'lr_mult': 10,
                'decay_mult': 0,
                'name': "addtional BN scale/shift"
            },
            {
                'params': base_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "base weight"
            },
            {
                'params': base_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "base bias"
            },
            {
                'params': base_bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "base BN scale/shift"
            },
        ]

    def forward(self, input):

        output, latent_output = self.encoder(input)
        prediction = self.decoder.forward(output)
        # latent_output = F.interpolate(latent_output, size=input.size()[2:], mode='bilinear', align_corners=True)
        return prediction, latent_output

