import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage import color
from skimage.color import lab2rgb, rgb2lab
from torchvision import models
from embadding_models import PretrainedModel1


device = "mps" if torch.backends.mps.is_available() else "cpu"

class SEBLock(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()

        self.app = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(C, C//r)
        self.linear2 = nn.Linear(C//r, C)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.app(x)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.sigmoid(out)

        out = out[:, :, None, None]

        res = out * x

        return res





class СolorizerV2(nn.Module):
    def __init__(self, inputs=1, outputs=2):
        super().__init__()

        self.conv1 = nn.Conv2d(inputs, 32, kernel_size=4, stride=2, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.seblock1 = SEBLock(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.seblock2 = SEBLock(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.seblock3 = SEBLock(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.seblock4 = SEBLock(256)

        self.btn_conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3, dilation=2)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.seblock5 = SEBLock(512)

        self.btn_conv6 = nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.bnorm6 = nn.BatchNorm2d(256)
        self.seblock6 = SEBLock(256)

        self.t_conv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.t_bnorm1 = nn.BatchNorm2d(128)
        self.seblock7 = SEBLock(128)

        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_bnorm2 = nn.BatchNorm2d(64)
        self.seblock8 = SEBLock(64)

        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_bnorm3 = nn.BatchNorm2d(32)
        self.seblock9 = SEBLock(32)

        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.t_conv5 = nn.Conv2d(3, outputs, kernel_size=3, padding=1)

    def forward(self, x):
        # print(x.shape)

        x1 = F.leaky_relu(self.bnorm1(self.conv1(x)))
        x1 = self.seblock1(x1)
        # print(x1.shape)

        x2 = F.leaky_relu(self.bnorm2(self.conv2(x1)))
        x2 = self.seblock2(x2)
        # print(x2.shape)

        x3 = F.leaky_relu(self.bnorm3(self.conv3(x2)))
        x3 = self.seblock3(x3)
        # print(x3.shape)

        x4 = F.leaky_relu(self.bnorm4(self.conv4(x3)))
        x4 = self.seblock4(x4)
        # print(x4.shape)

        x5 = F.leaky_relu(self.bnorm5(self.btn_conv5(x4)))
        x5 = self.seblock5(x5)
        # print(x5.shape)

        x6 = F.leaky_relu(self.bnorm6(self.btn_conv6(x5)))
        x6 = self.seblock6(x6)
        # print(x6.shape)

        x7 = torch.cat((x6, x4), 1)
        # print(x7.shape)

        x8 = F.leaky_relu(self.t_bnorm1(self.t_conv1(x7)))
        x8 = self.seblock7(x8)
        # print(x8.shape)

        x9 = torch.cat((x8, x3), 1)
        # print(x9.shape)

        x10 = F.leaky_relu(self.t_bnorm2(self.t_conv2(x9)))
        x10 = self.seblock8(x10)
        # print(x10.shape)

        x11 = torch.cat((x10, x2), 1)
        # print(x11.shape)

        x12 = F.leaky_relu(self.t_bnorm3(self.t_conv3(x11)))
        x12 = self.seblock9(x12)
        # print(x12.shape)

        x13 = torch.cat((x12, x1), 1)
        # print(x13.shape)

        x14 = F.leaky_relu(self.t_conv4(x13))
        # print(x14.shape)

        x15 = torch.cat((x14, x), 1)
        # print(x15.shape)

        x16 = F.sigmoid(self.t_conv5(x15))
        # print(x16.shape)

        return x16

    def predict(self, l_channel):
        self.eval()

        with torch.no_grad():
            ab_channels = self.forward(l_channel)

        l_channel = l_channel.squeeze(0).cpu().numpy()
        # print(l_channel.shape)
        a_channel = ab_channels[:, 0, :, :].cpu().numpy()
        b_channel = ab_channels[:, 1, :, :].cpu().numpy()

        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        # print(l_channel.shape, a_channel.shape, b_channel.shape)
        lab_img = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_img)

        # if first_way == True:
        rgb_image = (rgb_image * 255).astype(np.uint8)

        return rgb_image


class СolorizerV5(nn.Module):
    def __init__(self, inputs=1, outputs=2):
        super().__init__()

        self.conv1 = nn.Conv2d(inputs, 32, kernel_size=4, stride=2, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.seblock1 = SEBLock(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.seblock2 = SEBLock(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.seblock3 = SEBLock(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.seblock4 = SEBLock(256)

        self.btn_conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3, dilation=2)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.seblock5 = SEBLock(512)

        self.btn_conv6 = nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.bnorm6 = nn.BatchNorm2d(256)
        self.seblock6 = SEBLock(256)

        self.t_conv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.t_bnorm1 = nn.BatchNorm2d(128)
        self.seblock7 = SEBLock(128)

        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_bnorm2 = nn.BatchNorm2d(64)
        self.seblock8 = SEBLock(64)

        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_bnorm3 = nn.BatchNorm2d(32)
        self.seblock9 = SEBLock(32)

        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.t_conv5 = nn.Conv2d(3, outputs, kernel_size=3, padding=1)

    def forward(self, x):
        # print(x.shape)

        x1 = F.leaky_relu(self.bnorm1(self.conv1(x)))
        x1 = self.seblock1(x1)
        # print(x1.shape)

        x2 = F.leaky_relu(self.bnorm2(self.conv2(x1)))
        x2 = self.seblock2(x2)
        # print(x2.shape)

        x3 = F.leaky_relu(self.bnorm3(self.conv3(x2)))
        x3 = self.seblock3(x3)
        # print(x3.shape)

        x4 = F.leaky_relu(self.bnorm4(self.conv4(x3)))
        x4 = self.seblock4(x4)
        # print(x4.shape)

        x5 = F.leaky_relu(self.bnorm5(self.btn_conv5(x4)))
        x5 = self.seblock5(x5)
        # print(x5.shape)

        x6 = F.leaky_relu(self.bnorm6(self.btn_conv6(x5)))
        x6 = self.seblock6(x6)
        # print(x6.shape)

        x7 = torch.cat((x6, x4), 1)
        # print(x7.shape)

        x8 = F.leaky_relu(self.t_bnorm1(self.t_conv1(x7)))
        x8 = self.seblock7(x8)
        # print(x8.shape)

        x9 = torch.cat((x8, x3), 1)
        # print(x9.shape)

        x10 = F.leaky_relu(self.t_bnorm2(self.t_conv2(x9)))
        x10 = self.seblock8(x10)
        # print(x10.shape)

        x11 = torch.cat((x10, x2), 1)
        # print(x11.shape)

        x12 = F.leaky_relu(self.t_bnorm3(self.t_conv3(x11)))
        x12 = self.seblock9(x12)
        # print(x12.shape)

        x13 = torch.cat((x12, x1), 1)
        # print(x13.shape)

        x14 = F.leaky_relu(self.t_conv4(x13))
        # print(x14.shape)

        x15 = torch.cat((x14, x), 1)
        # print(x15.shape)

        x16 = F.tanh(self.t_conv5(x15))
        # print(x16.shape)

        return x16

    def predict(self, l_channel):
        self.eval()

        with torch.no_grad():
            ab_channels = self.forward(l_channel)

        l_channel = l_channel.squeeze(0).cpu().numpy()
        # print(l_channel.shape)
        a_channel = ab_channels[:, 0, :, :].cpu().numpy()
        b_channel = ab_channels[:, 1, :, :].cpu().numpy()


        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        # print(l_channel.shape, a_channel.shape, b_channel.shape)
        lab_img = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_img)

        # if first_way == True:
        rgb_image = (rgb_image * 255).astype(np.uint8)


        return rgb_image


class ColorizerV6(nn.Module):
    def __init__(self, inputs=1, outputs=2):
        super().__init__()

        self.conv1 = nn.Conv2d(inputs, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3, dilation=2)
        self.t_conv5 = nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=3, dilation=2)

        self.t_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(32, outputs, kernel_size=4, stride=2, padding=1)

        # self.t_conv = nn.ConvTranspose2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_latent=False):
        out1 = F.leaky_relu(self.conv1(x))
        # print(out1.shape)
        out2 = F.leaky_relu(self.conv2(out1))
        # print(out2.shape)
        out3 = F.leaky_relu(self.conv3(out2))
        # print(out3.shape)
        out4 = F.leaky_relu(self.conv4(out3))
        # print(out4.shape)

        out5 = F.leaky_relu(self.conv5(out4))
        # print(out5.shape)

        if return_latent == True:
            return out5

        out6 = F.leaky_relu(self.t_conv5(out5))
        # print(out6.shape)
        out7 = F.leaky_relu(self.t_conv4(out6))
        # print(out7.shape)
        out8 = F.leaky_relu(self.t_conv3(out7))
        # print(out8.shape)
        out9 = F.leaky_relu(self.t_conv2(out8))
        # print(out9.shape)
        out = F.tanh(self.t_conv1(out9))

        return out


    def predict(self, l_channel):
        self.eval()

        with torch.no_grad():
            ab_channels = self.forward(l_channel)

        l_channel = l_channel.squeeze(0).cpu().numpy()
        # print(l_channel.shape)
        a_channel = ab_channels[:, 0, :, :].cpu().numpy()
        b_channel = ab_channels[:, 1, :, :].cpu().numpy()

        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        # print(l_channel.shape, a_channel.shape, b_channel.shape)
        lab_img = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_img)

        # if first_way == True:
        rgb_image = (rgb_image * 255).astype(np.uint8)

        return rgb_image


# class SEBlock(nn.Module):
#     def __init__(self, C, r=16):
#         super().__init__()
#
#         self.aap = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#
#         self.linear1 = nn.Linear(C, C // r)
#         self.linear2 = nn.Linear(C // r, C)
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         out = self.aap(x)
#         out = self.flatten(out)
#
#         out = self.relu(self.linear1(out))
#         out = self.sigmoid(self.linear2(out))
#
#         out = out[:, :, None, None]
#
#         res = out * x
#         return res
#
#
# class SE_ResBlock(nn.Module):
#     def __init__(self, inputs, outputs, kernel, stride):
#         super().__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=1),
#             nn.BatchNorm2d(outputs),
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(outputs),
#         )
#
#         if inputs != outputs:
#             self.add_conv = nn.Sequential(
#                 nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=1),
#                 nn.BatchNorm2d(outputs),
#             )
#
#         self.se_block = SEBlock(outputs)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         add_out = self.add_conv(x)
#
#         out = F.leaky_relu(add_out)
#         out = self.conv2(out)
#         out = self.se_block(out)
#
#         out += add_out
#
#         out = F.leaky_relu(out)
#
#         return out


class SEBlock(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(C, C // r)
        self.linear2 = nn.Linear(C // r, C)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.aap(x)
        out = self.flatten(out)

        out = self.relu(self.linear1(out))
        out = self.sigmoid(self.linear2(out))

        out = out[:, :, None, None]

        res = out * x
        return res


class SE_ResBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(outputs),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outputs),
        )

        if inputs != outputs:
            self.add_conv = nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=1),
                nn.BatchNorm2d(outputs),
            )

        self.se_block = SEBlock(outputs)

    def forward(self, x):
        out = self.conv1(x)
        add_out = self.add_conv(x)

        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.se_block(out)

        out += add_out

        out = F.leaky_relu(out)

        return out


class ColorizerV7(nn.Module):
    def __init__(self, inputs=1, outputs=2):
        super().__init__()

        self.conv1 = SE_ResBlock(inputs, 32, kernel=4, stride=2)
        self.conv2 = SE_ResBlock(32, 64, kernel=4, stride=2)
        self.conv3 = SE_ResBlock(64, 128, kernel=4, stride=2)
        self.conv4 = SE_ResBlock(128, 256, kernel=4, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # self.emb = nn.Sequential(
        #     SE_ResBlock(1024, 512, kernel=3, stride=1),
        # )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.t_conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.t_conv5 = nn.ConvTranspose2d(3, outputs, kernel_size=3, stride=1, padding=1)

        self.linear_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(2048, 512)
        )

        # ----------------------------------------------------------------------------------------------------------------

        self.conv2_1 = SE_ResBlock(inputs, 32, kernel=4, stride=2)
        self.conv2_2 = SE_ResBlock(32, 64, kernel=4, stride=2)
        self.conv2_3 = SE_ResBlock(64, 128, kernel=4, stride=2)
        self.conv2_4 = SE_ResBlock(128, 256, kernel=4, stride=2)

        self.conv2_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # self.emb = nn.Sequential(
        #     SE_ResBlock(1024, 512, kernel=3, stride=1),
        # )

        self.conv2_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t_conv2_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t_conv2_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.t_conv2_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.t_conv2_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.t_conv2_5 = nn.ConvTranspose2d(3, outputs, kernel_size=3, stride=1, padding=1)

        # ------------------------------------------------------------------------------------

        self.conv3_1 = SE_ResBlock(4, 32, kernel=4, stride=2)
        self.conv3_2 = SE_ResBlock(32, 64, kernel=4, stride=2)
        self.conv3_3 = SE_ResBlock(64, 128, kernel=4, stride=2)
        self.conv3_4 = SE_ResBlock(128, 256, kernel=4, stride=2)

        self.conv3_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.conv3_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t_conv3_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t_conv3_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.t_conv3_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.t_conv3_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.t_conv3_5 = nn.ConvTranspose2d(6, outputs, kernel_size=3, stride=1, padding=1)

    def forward(self, x, embadding=None, return_latent=False):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.conv5(x4)

        if embadding is not None:
            embadding_tensor = self.linear_block(embadding)

            embadding_tensor = embadding_tensor.unsqueeze(-1).unsqueeze(-1)
            embadding_tensor = embadding_tensor.expand(-1, -1, x5.size(2), x5.size(3))
        else:
            print("No embedding provided.")
            embadding_tensor = torch.zeros(x5.size(0), 512, x5.size(2), x5.size(3)).to(device)

        # x5 = torch.concat((embadding_tensor, x5), dim=1)
        # x5 = self.emb(x5)

        x5 += embadding_tensor

        if return_latent == True:
            return x5

        x6 = self.conv6(x5)

        x7 = torch.cat((x6, x4), 1)
        x7 = self.t_conv1(x7)

        x8 = torch.cat((x7, x3), 1)
        x8 = self.t_conv2(x8)

        x9 = torch.cat((x8, x2), 1)
        x9 = self.t_conv3(x9)

        x10 = torch.cat((x9, x1), 1)
        x10 = self.t_conv4(x10)

        x11 = torch.cat((x10, x), 1)
        x11 = F.tanh(self.t_conv5(x11))

        # -----------------------------------------------

        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x2_1)
        x2_3 = self.conv2_3(x2_2)
        x2_4 = self.conv2_4(x2_3)

        x2_5 = self.conv2_5(x2_4)

        x2_6 = self.conv2_6(x2_5)

        x2_7 = torch.cat((x2_6, x2_4), 1)
        x2_7 = self.t_conv2_1(x2_7)

        x2_8 = torch.cat((x2_7, x2_3), 1)
        x2_8 = self.t_conv2_2(x2_8)

        x2_9 = torch.cat((x2_8, x2_2), 1)
        x2_9 = self.t_conv2_3(x2_9)

        x2_10 = torch.cat((x2_9, x2_1), 1)
        x2_10 = self.t_conv2_4(x2_10)

        x2_11 = torch.cat((x2_10, x), 1)
        x2_11 = F.tanh(self.t_conv2_5(x2_11))

        x3 = torch.cat((x11, x2_11), 1)

        # ------------------------------------------------------

        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_2(x3_1)
        x3_3 = self.conv3_3(x3_2)
        x3_4 = self.conv3_4(x3_3)

        x3_5 = self.conv3_5(x3_4)

        x3_6 = self.conv3_6(x3_5)

        x3_7 = torch.cat((x3_6, x3_4), 1)
        x3_7 = self.t_conv3_1(x3_7)

        x3_8 = torch.cat((x3_7, x3_3), 1)
        x3_8 = self.t_conv3_2(x3_8)

        x3_9 = torch.cat((x3_8, x3_2), 1)
        x3_9 = self.t_conv3_3(x3_9)

        x3_10 = torch.cat((x3_9, x3_1), 1)
        x3_10 = self.t_conv3_4(x3_10)

        x3_11 = torch.cat((x3_10, x3), 1)
        res = F.tanh(self.t_conv3_5(x3_11))

        return res

    def predict(self, l_channel, rgb_image=None):
        self.eval()

        with torch.no_grad():
            if rgb_image is not None:
                ab_channels = self.forward(l_channel, embadding=rgb_image)
            else:
                ab_channels = self.forward(l_channel)

        l_channel = l_channel.squeeze(0).permute(1, 2, 0).squeeze(2).cpu().numpy()
        # l_channel =  l_channel.permute(1, 2, 0).cpu().numpy()
        ab_channels = ab_channels.squeeze(0).permute(1, 2, 0).cpu().numpy()

        a_channel = ab_channels[:, :, 0]
        b_channel = ab_channels[:, :, 1]

        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        lab_image = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_image)

        rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

        # rgb_image =  np.clip(rgb_image, 0, 1).astype(np.uint8)

        return rgb_image


class ColorizerV8(nn.Module):
    def __init__(self, inputs=1, outputs=2):
        super().__init__()

        self.conv1 = SE_ResBlock(inputs, 32, kernel=4, stride=2)
        self.conv2 = SE_ResBlock(32, 64, kernel=4, stride=2)
        self.conv3 = SE_ResBlock(64, 128, kernel=4, stride=2)
        self.conv4 = SE_ResBlock(128, 256, kernel=4, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # self.emb = nn.Sequential(
        #     SE_ResBlock(1024, 512, kernel=3, stride=1),
        # )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.t_conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.t_conv5 = nn.ConvTranspose2d(3, outputs, kernel_size=3, stride=1, padding=1)

        # self.linear_block = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Linear(emb_size, 512)
        # )

        # ----------------------------------------------------------------------------------------------------------------

        self.conv2_1 = SE_ResBlock(inputs, 32, kernel=4, stride=2)
        self.conv2_2 = SE_ResBlock(32, 64, kernel=4, stride=2)
        self.conv2_3 = SE_ResBlock(64, 128, kernel=4, stride=2)
        self.conv2_4 = SE_ResBlock(128, 256, kernel=4, stride=2)

        self.conv2_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # self.emb = nn.Sequential(
        #     SE_ResBlock(1024, 512, kernel=3, stride=1),
        # )

        self.conv2_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t_conv2_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t_conv2_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.t_conv2_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.t_conv2_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.t_conv2_5 = nn.ConvTranspose2d(3, outputs, kernel_size=3, stride=1, padding=1)

        # ------------------------------------------------------------------------------------

        self.conv3_1 = SE_ResBlock(4, 32, kernel=4, stride=2)
        self.conv3_2 = SE_ResBlock(32, 64, kernel=4, stride=2)
        self.conv3_3 = SE_ResBlock(64, 128, kernel=4, stride=2)
        self.conv3_4 = SE_ResBlock(128, 256, kernel=4, stride=2)

        self.conv3_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.conv3_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t_conv3_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t_conv3_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.t_conv3_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.t_conv3_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.t_conv3_5 = nn.ConvTranspose2d(6, outputs, kernel_size=3, stride=1, padding=1)

    def forward(self, x, embadding=None, return_latent=False):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.conv5(x4)

        if embadding is not None:
            # if embadding.size(1) != 512:
            #     embadding = self.linear_block(embadding)

            embadding_tensor = embadding.unsqueeze(-1).unsqueeze(-1)
            embadding_tensor = embadding_tensor.expand(-1, -1, x5.size(2), x5.size(3))
        else:
            print("No embedding provided.")
            embadding_tensor = torch.zeros(x5.size(0), 512, x5.size(2), x5.size(3)).to(device)

        # x5 = torch.concat((embadding_tensor, x5), dim=1)
        # x5 = self.emb(x5)

        x5 += embadding_tensor

        if return_latent == True:
            return x5

        x6 = self.conv6(x5)

        x7 = torch.cat((x6, x4), 1)
        x7 = self.t_conv1(x7)

        x8 = torch.cat((x7, x3), 1)
        x8 = self.t_conv2(x8)

        x9 = torch.cat((x8, x2), 1)
        x9 = self.t_conv3(x9)

        x10 = torch.cat((x9, x1), 1)
        x10 = self.t_conv4(x10)

        x11 = torch.cat((x10, x), 1)
        x11 = F.tanh(self.t_conv5(x11))

        # -----------------------------------------------

        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x2_1)
        x2_3 = self.conv2_3(x2_2)
        x2_4 = self.conv2_4(x2_3)

        x2_5 = self.conv2_5(x2_4)

        x2_6 = self.conv2_6(x2_5)

        x2_7 = torch.cat((x2_6, x2_4), 1)
        x2_7 = self.t_conv2_1(x2_7)

        x2_8 = torch.cat((x2_7, x2_3), 1)
        x2_8 = self.t_conv2_2(x2_8)

        x2_9 = torch.cat((x2_8, x2_2), 1)
        x2_9 = self.t_conv2_3(x2_9)

        x2_10 = torch.cat((x2_9, x2_1), 1)
        x2_10 = self.t_conv2_4(x2_10)

        x2_11 = torch.cat((x2_10, x), 1)
        x2_11 = F.tanh(self.t_conv2_5(x2_11))

        x3 = torch.cat((x11, x2_11), 1)

        # ------------------------------------------------------

        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_2(x3_1)
        x3_3 = self.conv3_3(x3_2)
        x3_4 = self.conv3_4(x3_3)

        x3_5 = self.conv3_5(x3_4)

        x3_6 = self.conv3_6(x3_5)

        x3_7 = torch.cat((x3_6, x3_4), 1)
        x3_7 = self.t_conv3_1(x3_7)

        x3_8 = torch.cat((x3_7, x3_3), 1)
        x3_8 = self.t_conv3_2(x3_8)

        x3_9 = torch.cat((x3_8, x3_2), 1)
        x3_9 = self.t_conv3_3(x3_9)

        x3_10 = torch.cat((x3_9, x3_1), 1)
        x3_10 = self.t_conv3_4(x3_10)

        x3_11 = torch.cat((x3_10, x3), 1)
        res = F.tanh(self.t_conv3_5(x3_11))

        return res

    def predict(self, l_channel, rgb_image=None):
        self.eval()

        with torch.no_grad():
            if rgb_image is not None:
                ab_channels = self.forward(l_channel, embadding=rgb_image)
            else:
                ab_channels = self.forward(l_channel)

        l_channel = l_channel.squeeze(0).permute(1, 2, 0).squeeze(2).cpu().numpy()
        # l_channel =  l_channel.permute(1, 2, 0).cpu().numpy()
        ab_channels = ab_channels.squeeze(0).permute(1, 2, 0).cpu().numpy()

        a_channel = ab_channels[:, :, 0]
        b_channel = ab_channels[:, :, 1]

        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        lab_image = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_image)

        rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

        # rgb_image =  np.clip(rgb_image, 0, 1).astype(np.uint8)

        return rgb_image


class EmbeddingExtractorV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.inception_v3(pretrained=True, aux_logits=True)
        self.pretrained_model.fc = nn.Identity()
    def forward(self, x):
        self.pretrained_model.eval()
        res = self.pretrained_model(x)
        return res


class EMB_SE_ResBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(outputs),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(outputs, outputs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outputs),
        )

        self.add_conv = nn.Sequential(
            nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(outputs),
        )

        self.se_block = SEBlock(outputs)

    def forward(self, x):
        out = self.conv1(x)
        add_out = self.add_conv(x)

        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.se_block(out)

        out += add_out

        out = F.leaky_relu(out)

        return out


class EmbeddingExtractorV2(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.swin = models.swin_t(pretrained=True)
        self.swin.head = nn.Linear(768, embedding_dim)

        self.linear = nn.Linear(512, 1024)
        self.conv1 = EMB_SE_ResBlock(1024, 512, kernel=4, stride=2)

        self.conv2 = EMB_SE_ResBlock(512, 256, kernel=4, stride=2)

        self.conv3 = EMB_SE_ResBlock(256, 256, kernel=4, stride=2)

        self.conv4 = EMB_SE_ResBlock(256, 128, kernel=4, stride=2)

        self.conv5 = EMB_SE_ResBlock(128, 64, kernel=4, stride=2)

        self.conv6 = EMB_SE_ResBlock(64, 32, kernel=4, stride=2)

        self.conv7 = EMB_SE_ResBlock(32, 16, kernel=4, stride=2)

        self.conv8 = EMB_SE_ResBlock(16, 8, kernel=4, stride=2)

        self.conv9 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_emb=False):
        x = self.swin(x)

        if return_emb == True:
            return x

        x = F.leaky_relu(self.linear(x))
        x = x.view(x.size(0), 1024, 1, 1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.conv6(x)
        # print(x.shape)
        x = self.conv7(x)
        # print(x.shape)
        x = self.conv8(x)
        x = F.sigmoid(self.conv9(x))

        return x


class SE_ResBlock2(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride, padding):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(outputs),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(outputs, outputs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outputs),
        )

        self.add_conv = nn.Sequential(
            nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(outputs),
        )

        self.se_block = SEBlock(outputs)


class EmbeddingExtractorV3(nn.Module):
    def __init__(self, embedding_dim=2048):
        super().__init__()
        # self.swin = models.swin_t(pretrained=True)
        # self.swin.head = nn.Linear(768, embedding_dim)

        self.basic_model = models.inception_v3(pretrained=True, aux_logits=True)
        # self.basic_model.AuxLogits = None

        num_features = self.basic_model.fc.in_features
        self.basic_model.fc = nn.Identity()

        self.linear = nn.Linear(num_features, 1024)
        self.conv1 = SE_ResBlock2(1024, 512, kernel=4, stride=2, padding=1)

        self.conv2 = SE_ResBlock2(512, 256, kernel=4, stride=2, padding=1)

        self.conv3 = SE_ResBlock2(256, 256, kernel=4, stride=2, padding=1)

        self.conv4 = SE_ResBlock2(256, 128, kernel=4, stride=2, padding=1)

        self.conv5 = SE_ResBlock2(128, 64, kernel=4, stride=2, padding=1)

        self.conv6 = SE_ResBlock2(64, 32, kernel=5, stride=3, padding=1)

        self.conv7 = SE_ResBlock2(32, 16, kernel=4, stride=2, padding=1)

        self.conv8 = SE_ResBlock2(16, 8, kernel=4, stride=2, padding=0)

        self.conv9 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=0)

    def forward(self, x, return_emb=False):
        out = self.basic_model(x)
        # out = out.logits

        if isinstance(out, tuple):
            out = out.logits

        if return_emb == True:
            return out

        # print(out.shape)
        out = F.leaky_relu(self.linear(out))
        # print(out.shape)
        # out = out.view(x.size(0), 1024, 1, 1)
        out = out.unsqueeze(2).unsqueeze(3)
        # print(x.shape)
        out = self.conv1(out)
        # print(x.shape)
        out = self.conv2(out)
        # print(x.shape)
        out = self.conv3(out)
        # print(x.shape)
        out = self.conv4(out)
        # print(x.shape)
        out = self.conv5(out)
        # print(x.shape)
        out = self.conv6(out)
        # print(x.shape)
        out = self.conv7(out)
        # print(x.shape)
        out = self.conv8(out)
        out = F.sigmoid(self.conv9(out))

        return out








def additional_model():
    model = СolorizerV2()
    model.load_state_dict(torch.load("weights/colorizer_wrong.pt", map_location=device))
    return model

def teacher_model():
    model = СolorizerV5()
    model.load_state_dict(torch.load("weights/colorizer_l.pt", map_location=device))
    return model

def student_model():
    model = ColorizerV6()
    model.load_state_dict(torch.load("weights/colorizer_s.pt", map_location=device))
    return model

# for 2048 as embedding
def emb_model_1():
    model = ColorizerV7()
    model.load_state_dict(torch.load("weights/history_in_color_v16.pt", map_location=device))
    return model

# for 512 as embedding
def emb_model_2():
    model = ColorizerV8()
    model.load_state_dict(torch.load("weights/history_in_color_v20.pt", map_location=device))
    return model


# pretrained inception (emb - 2048)
def extractor_1():
    model = EmbeddingExtractorV1()
    return model

# custom swin (emb - 512)
def extractor_2():
    model = EmbeddingExtractorV2()
    model.load_state_dict(torch.load("weights/embedding_extractor_1.pt", map_location=device))
    return model

# custom inception (emb - 2048)
def extractor_3():
    model = EmbeddingExtractorV3()
    model.load_state_dict(torch.load("weights/embedding_extractor_2.pt", map_location=device))
    return model

# model = emb_model_1()
# model.to(device)
#
#
# tensor = torch.rand(1, 1, 256, 256).to(device)
# embadding_tensor = torch.rand(1, 2048).to(device)
# res = model.predict(tensor, embadding_tensor)
# print(res.shape)


model = emb_model_1()
emb_model = extractor_3()

model.to(device)
emb_model.to(device)

tensor = torch.rand(1, 1, 256, 256).to(device)
emb_tensor = torch.rand(1, 3, 384, 384).to(device)

embedding = emb_model(emb_tensor, return_emb=True)
res = model(tensor, embedding)
# emb = torch.rand(1, 2048).to(device)
print(res.shape)









