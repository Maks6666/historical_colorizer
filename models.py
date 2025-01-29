import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage import color
from skimage.color import lab2rgb, rgb2lab


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

        # print("Shape of output from model:", ab_channels.shape)
        # print("Min/Max values in ab_channels (before scaling):", ab_channels.min().item(), ab_channels.max().item())

        l_channel = l_channel.squeeze(0).cpu().numpy()
        a_channel = ab_channels[:, 0, :, :].cpu().numpy()
        b_channel = ab_channels[:, 1, :, :].cpu().numpy()

        # print("Min/Max a_channel (after extraction):", a_channel.min(), a_channel.max())
        # print("Min/Max b_channel (after extraction):", b_channel.min(), b_channel.max())

        # Масштабируем значения
        l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel * 128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)

        lab_img = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_img)

        # if first_way == True:
        rgb_image = (rgb_image * 255).astype(np.uint8)
        # # rgb_image = np.clip(rgb_image, 0, 255)
        # elif second_way == True:
        #     rgb_image = rgb_image * 255
        #     rgb_image = np.clip(rgb_image, 0, 255)

        # print("Min/Max RGB image:", rgb_image.min(), rgb_image.max())

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
            ab_channel = self.forward(l_channel)

        L = l_channel.squeeze(0).squeeze(0).cpu().numpy()
        AB = ab_channel.squeeze(0).cpu().numpy()

        A = AB[0, :, :]
        B = AB[1, :, :]

        L = np.clip(L * 100, 0, 100).astype(np.uint8)
        A = np.clip(A * 128, -128, 127).astype(np.int8)
        B = np.clip(B * 128, -128, 127).astype(np.int8)

        lab_image = np.stack([L, A, B], axis=-1).astype(np.float64)
        rgb_image = lab2rgb(lab_image)
        rgb_image = (rgb_image * 255).astype(np.uint8)

        return rgb_image




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

model = additional_model()
tensor = torch.rand(1, 1, 256, 256)
res = model.predict(tensor)
print(res.shape)










