from torch.utils.data import Dataset
from skimage.color import rgb2lab
import torch

class CustomVideoDataset(Dataset):
    def __init__(self, tesnor_list):
        self.tesnor_list = tesnor_list
    def __len__(self):
        return len(self.tesnor_list)
    def __getitem__(self, idx):
        img = self.tesnor_list[idx]
        image = img.permute(1, 2, 0).cpu().numpy()
        lab_image = rgb2lab(image)
        l_channel = lab_image[:, :, 0] / 100.0

        l_channel = torch.tensor(l_channel, dtype=torch.float32).unsqueeze(0)

        return l_channel