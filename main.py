from os import listdir
import time
from skimage.color import rgb2lab
from models import teacher_model, student_model, additional_model
import torch
from PIL import Image
import os
import numpy as np
from skimage import color
from torchvision import transforms
import cv2

from tools import choose_file


device = "mps" if torch.backends.mps.is_available() else "cpu"


class ColorizerApp:
    def __init__(self, image, dir_to_save, model_marker):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dir_to_save = dir_to_save
        self.model_marker = model_marker
        self.model = self.load_model()
        self.image = image
        self.transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor()
        ])




    def load_model(self):
        if self.model_marker == "s":
            return student_model()
        elif self.model_marker == "l":
            return teacher_model()
        elif self.model_marker == "A":
            return additional_model()


    def output_result(self):
        idx = len(os.listdir(self.dir_to_save))+1
        os.makedirs(self.dir_to_save, exist_ok=True)

        start = time.time()

        if os.path.isfile(self.image):

            image = Image.open(self.image).convert('RGB')

            width, height = image.size

            image = self.transforms(image)
            image = image.permute(1, 2, 0).cpu().numpy()
            print(image.shape)
            lab_img = color.rgb2lab(image)

            l_channel = lab_img[:, :, 0] / 100.0

            l_channel = torch.tensor(l_channel, dtype=torch.float32).unsqueeze(2)

            l_channel = l_channel.permute(2, 0, 1)
            l_channel = l_channel.unsqueeze(0).to(self.device)

            self.model.to(self.device)
            res = self.model.predict(l_channel)

            if len(res.shape) == 4:
                res = res.squeeze(0)

            if np.max(res) <= 1:
                res = (res * 255).astype(np.uint8)

            res = cv2.resize(res, (width, height))
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            path = os.path.join(self.dir_to_save, f"{idx}.jpg")

            # l_channel = np.clip(l_channel * 255, 0, 255).astype(np.uint8)

            cv2.imwrite(path, res)

            end = time.time()

            print(f"Done! Proces took: {end-start:.2f}s")

            return res

# model = first_model()

root_dir = "grayscaled_images"
image = choose_file(root_dir)
dir_to_save = "colorized_images"


colorizer = ColorizerApp(image, dir_to_save, "l")
image = colorizer.output_result()


