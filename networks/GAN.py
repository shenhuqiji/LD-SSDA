# camera-ready

import torch.nn as nn
import torch.nn.functional as F
import torch

class OutputDiscriminator(nn.Module):
    def __init__(self, ):
        super(OutputDiscriminator, self).__init__()

        filter_num_list = [64, 128, 256, 512, 1]

        self.conv1 = nn.Conv2d(2, filter_num_list[0], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=4, stride=2, padding=2, bias=False)
        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=4, stride=2, padding=2, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self,):
        super(DomainDiscriminator, self).__init__()

        self.F = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=2, bias=False),
        )
        self.D = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=2, bias=False),
		)

    def forward(self, x1, x2):
        x1 = self.F(x1)
        x2 = self.F(x2)
        concat = torch.cat((x1, x2), dim=1)
        out = self.D(concat)

        return out

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(2048, 512)
        self.linear4 = nn.Linear(512, 2048)
        self.LayerNorm1 = nn.LayerNorm(256, eps=1e-6)
        self.LayerNorm2 = nn.LayerNorm(2048, eps=1e-6)

    def forward(self, input):
    ### sequence mlp _c+s
        output = input.flatten(2).transpose(-1, -2)
        output_c = self.linear1(self.LayerNorm1(output))
        output_c = F.gelu(output_c)
        output_c = self.linear2(output_c)
        output_c = output_c + output
        output_c = output_c.permute(0, 2, 1)

        output1 = input.flatten(2)
        output_s = self.linear3(self.LayerNorm2(output1))
        output_s = F.gelu(output_s)
        output_s = self.linear4(output_s)
        output_s = output_s + output1

        output_all = output_s + output_c

        return output_all


class DomainDiscriminator_MLP(nn.Module):
    def __init__(self,):
        super(DomainDiscriminator_MLP, self).__init__()

        self.Mlp = Mlp()
        self.fc1 = nn.Linear(512, 1)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        # self.fc1 = nn.Linear(4096, 512)
        # self.fc2 = nn.Linear(512, 128)
        self.LayerNorm1 = nn.LayerNorm(512, eps=1e-6)
        self.LayerNorm2 = nn.LayerNorm(128, eps=1e-6)

    def forward(self, x1, x2):
        x1 = self.Mlp(x1)
        x2 = self.Mlp(x2)
        concat = torch.cat((x1, x2), dim=1)
        concat = concat.transpose(-1, -2)
        fc1 = self.fc1(concat).permute(0, 2, 1)
        fc2 = self.fc2(fc1)
        fc2 = self.LayerNorm1(fc2)
        fc3 =  self.fc3(fc2)
        out = self.LayerNorm2(fc3)
        # concat = torch.cat((x1, x2), dim=2)
        # fc1 = self.fc1(concat)
        # fc1 = self.LayerNorm1(fc1)
        # fc2 = self.fc2(fc1)
        # out = self.LayerNorm2(fc2)
        return out





# class DomainDiscriminator(nn.Module):
#     def __init__(self,):
#         super(DomainDiscriminator, self).__init__()
#
#         self.conv = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1, bias=False)
#         self.D = nn.Sequential(
#             nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=2, bias=False),
# 		)
#
#     def forward(self, x1, x2):
#         concat = torch.cat((x1, x2), dim=1)
#         concat = self.conv(concat)
#         out = self.D(concat)
#
#         return out


# class DomainDiscriminator(nn.Module):
#     def __init__(self,):
#         super(DomainDiscriminator, self).__init__()
#
#         self.D = nn.Sequential(
#             nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=2, bias=False),
#             nn.LeakyReLU(negative_slope=0.2)
# 		)
#         self.out = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=2, bias=False)
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0.0, 0.02)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#
#     def forward(self, x1, x2):
#         src_out = self.D(x1)
#         tgt_out = self.D(x2)
#         concat = torch.cat((src_out, tgt_out), dim=1)
#         out = self.out(concat)
#
#         return out
