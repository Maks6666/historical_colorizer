from torch.utils.data import Dataset
from skimage.color import rgb2lab
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms

image_transformer = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

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



def custom_image_transformer(link, transformer=image_transformer):

    image = Image.open(link).convert('RGB')
    gray_rgb_image = image.convert('L').convert("RGB")

    if transformer:
        image = transformer(image)
        gray_rgb_image = transformer(gray_rgb_image)

    image = image.permute(1, 2, 0).cpu().numpy()
    lab_image = rgb2lab(image)
    l_channel = lab_image[:, :, 0] / 100.0

    l_channel = torch.tensor(l_channel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    gray_rgb_image =gray_rgb_image.unsqueeze(0)

    return l_channel, gray_rgb_image





link = "grayscaled_images/2.jpg"
gray_rgb_image, l_channel = custom_image_transformer(link)
print(gray_rgb_image.shape, l_channel.shape)