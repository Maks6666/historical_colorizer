import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage import color


device = "mps" if torch.backends.mps.is_available() else "cpu"

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(channels, channels // r)
        self.linear2 = nn.Linear(channels // r, channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.aap(x)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.sigmoid(out)

        out = out[:, :, None, None]

        res = out * x

        return res


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



# -------------------------------------------------------------------------------------------

class ColorizerV1(nn.Module):
    def __init__(self):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.SE_block2 = SEBlock(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.SE_block3 = SEBlock(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.SE_block4 = SEBlock(256)

        # Dilation layers.
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv5_bn = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(128)

        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(64)
        self.t_SE_block_2 = SEBlock(64)

        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(32)
        self.t_SE_block_3 = SEBlock(32)

        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # print(x.shape)
        x_1 = F.relu(self.conv1_bn(self.conv1(x)))
        # print(x_1.shape)
        x_2 = F.relu(self.conv2_bn(self.conv2(x_1)))
        x_2 = self.SE_block2(x_2)
        # print(x_2.shape)
        x_3 = F.relu(self.conv3_bn(self.conv3(x_2)))
        x_3 = self.SE_block3(x_3)
        # print(x_3.shape)
        x_4 = F.relu(self.conv4_bn(self.conv4(x_3)))
        x_4 = self.SE_block4(x_4)
        # print(x_4.shape)

        # Dilation layers.
        x_5 = F.relu(self.conv5_bn(self.conv5(x_4)))
        # x_5 = self.dropout(x_5)
        x_5_d = F.relu(self.conv6_bn(self.conv6(x_5)))
        # print(x_5_d.shape)

        x_6 = F.relu(self.t_conv1_bn(self.t_conv1(x_5_d)))
        # print(x_6.shape)
        x_6 = torch.cat((x_6, x_3), 1)
        # print(x_6.shape)
        x_7 = F.relu(self.t_conv2_bn(self.t_conv2(x_6)))
        x_7 = self.t_SE_block_2(x_7)
        # print(x_7.shape)

        x_7 = torch.cat((x_7, x_2), 1)
        # print(x_7.shape)

        x_8 = F.relu(self.t_conv3_bn(self.t_conv3(x_7)))
        x_8 = self.t_SE_block_3(x_8)
        # print(x_8.shape)
        x_8 = torch.cat((x_8, x_1), 1)
        # print(x_8.shape)
        x_9 = F.relu(self.t_conv4(x_8))
        # print(x_9.shape)
        # here we concatenate input tensor with size 1xHxW and previouse layer output with size of 2xHxW
        x_9 = torch.cat((x_9, x), 1)
        # print(x_9.shape)
        x = self.output(x_9)
        # print(x.shape)
        return x

    def predict(self, l_channel):
        self.eval()

        with torch.no_grad():
            ab_channels = self.forward(l_channel)

        l_channel = l_channel.squeeze(0).cpu().numpy()
        a_channel = ab_channels[:, 0, :, :].cpu().numpy()
        b_channel = ab_channels[:, 1, :, :].cpu().numpy()

        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        lab_img = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float32)
        rgb_img = color.lab2rgb(lab_img)

        rgb_img = np.clip(rgb_img, 0, 1)

        rgb_img = rgb_img.squeeze(0)

        return rgb_img

 # -------------------------------------------------------------------------------------------


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

        a_channel = ab_channels[:, 0, :, :].cpu().numpy()
        b_channel = ab_channels[:, 1, :, :].cpu().numpy()

        # print(a_channel.shape, b_channel.shape, l_channel.shape)
        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        lab_img = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float32)
        rgb_image = color.lab2rgb(lab_img)

        rgb_image = np.clip(rgb_image, 0, 1)

        rgb_image = rgb_image.squeeze(0)

        return rgb_image


def first_model():
    model = ColorizerV1()
    model.load_state_dict(torch.load("weights/autoencoder_colorizer_v03.pt", map_location=device))
    return model


def second_model():
    model = СolorizerV2()
    model.load_state_dict(torch.load("weights/history_in_color_v03.pt", map_location=device))
    return model

model = second_model()








