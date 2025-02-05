from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F

class PretrainedModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.convnext_large(pretrained=True)
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(1000, 512)


    def forward(self, x):
        out = self.model(x)
        out = F.leaky_relu(self.linear(out))
        return out

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


model = PretrainedModel1()
tensor = torch.rand(1, 3, 256, 256)

res = model.predict(tensor)
# print(res.shape)