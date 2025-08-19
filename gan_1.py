import torch
from torch import nn
import torch.nn.functional as F
from skimage import color
import warnings

from blocks import SEBLock, ConvBlock, DeConvBlock
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # 1, 128, 128
        self.conv1 = ConvBlock(1, 64, 4, 2, 1, True, False)
        self.se1 = SEBLock(64)
        # 64, 64, 64
        self.conv2 = ConvBlock(64, 64, 4, 2, 1, True, False)
        self.se2 = SEBLock(64)
        # 64, 32, 32
        self.conv3 = ConvBlock(64, 128, 4, 2, 1, True, False)
        self.se3 = SEBLock(128)
        # 128, 16, 16
        self.conv4 = ConvBlock(128, 256, 4, 2, 1, True, False)
        self.se4 = SEBLock(256)
        # 128, 8, 8
        self.conv5 = ConvBlock(256, 512, 4, 2, 1, True, False)
        self.se5 = SEBLock(512)
        # 512, 4, 4
        self.bottle_neck = nn.Conv2d(512, 512, 4, 1, 3, dilation=2)
        self.se6 = SEBLock(512)
        # 256, 4, 4

        self.up_conv1 = DeConvBlock(1024, 256, 4, 2, 1, True)
        self.se7 = SEBLock(256)
         # 256, 4, 4

        self.up_conv2 = DeConvBlock(512, 128, 4, 2, 1, True)
        self.se8 = SEBLock(128)

        self.up_conv3 = DeConvBlock(256, 64, 4, 2, 1, True)
        self.se9 = SEBLock(64)

        self.up_conv4 = DeConvBlock(128, 64, 4, 2, 1, True)
        self.se10 = SEBLock(64)

        self.up_conv5 = DeConvBlock(128, 2, 4, 2, 1, True)

        self.up_conv6 = nn.Conv2d(3, 2, 3, 1, 1)



    def forward(self, x):
        out1 = self.se1(self.conv1(x))

        out2 = self.se2(self.conv2(out1))

        out3 = self.se3(self.conv3(out2))

        out4 = self.se4(self.conv4(out3))

        out5 = self.se5(self.conv5(out4))

        b = self.se6(self.bottle_neck(out5))

        d_out_1 = torch.cat((b, out5), dim=1)
        d_out_2 = self.se7(self.up_conv1(d_out_1))

        d_out_2 = torch.cat((d_out_2, out4), dim=1)
        d_out_3 = self.se8(self.up_conv2(d_out_2))

        d_out_3 = torch.cat((d_out_3, out3), dim=1)
        d_out_4 = self.se9(self.up_conv3(d_out_3))

        d_out_4 = torch.cat((d_out_4, out2), dim=1)
        d_out_5 = self.se10(self.up_conv4(d_out_4))

        d_out_5 = torch.cat((d_out_5, out1), dim=1)
        d_out_6 = self.up_conv5(d_out_5)

        out = torch.cat((d_out_6, x), dim=1)
        out = self.up_conv6(out)

        out = torch.tanh(out)


        return out

    def predict(self, l):

        if len(l.shape) == 3:
            l = l.unsqueeze(0)

        self.eval()

        with torch.no_grad():
            ab = self.forward(l)

        l_channel = l.squeeze(0).cpu().numpy()
        a_channel = ab[:, 0, :, :].cpu().numpy()
        b_channel = ab[:, 1, :, :].cpu().numpy()

        l_channel = np.clip(l_channel*100, 0, 100).astype(np.uint8)
        a_channel = np.clip(a_channel*128, -128, 127).astype(np.int8)
        b_channel = np.clip(b_channel*128, -128, 127).astype(np.int8)

        # print(l_channel.shape, a_channel.shape, b_channel.shape)


        lab_image = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)


        rgb_image = color.lab2rgb(lab_image)

        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_image = rgb_image.squeeze(0)


        return rgb_image

model = Generator()
weights = "weights/generator.pt"
device = "mps" if torch.backends.mps.is_available() else "cpu"

warnings.filterwarnings("ignore", category=FutureWarning)
model.load_state_dict(torch.load(weights, map_location=device))
model.to(device)
print("All keys matched!")



