from torch.utils.data import dataset
from skimage.color import rgb2lab
import torch

class VideoDataset(dataset.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        img = self.data_list[idx]
        image = img.permute(1, 2, 0).cpu().numpy()
        lab_image = rgb2lab(image)
        l_channels = lab_image[:, :, 0] / 100.0

        l_channels = torch.tensor(l_channels).unsqueeze(0)

        return l_channels
