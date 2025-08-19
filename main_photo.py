from gan_1 import model
import torch
import cv2
from PIL import Image
import numpy as np
from skimage import color
from torchvision import transforms

import time
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"


class Colorizer:
    def __init__(self, model, base_dir, dir_to_save):
        self.model = model
        self.transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
        ])

        self.base_dir = base_dir
        self.device = device
        self.dir_to_save = dir_to_save

    def handle_img(self, img):
        img_link = os.path.join(self.base_dir, img)
        img = Image.open(img_link).convert('RGB')
        h, w = img.size
        transformed_img = self.transformer(img)

        img = transformed_img.permute(1, 2, 0)
        img = img.cpu().numpy()

        lab_img = color.rgb2lab(img)

        l_channel = lab_img[:, :, 0] / 100.0

        l_channel = torch.tensor(l_channel, dtype=torch.float).unsqueeze(0).to(self.device)
        return h, w, l_channel

    def save_img(self, res, h, w, file_name_to_save):
        if np.max(res) <= 1:
            res = (res * 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        res = cv2.resize(res, (h, w))

        file_name = os.path.join(self.dir_to_save,  "colorized_" + file_name_to_save)
        cv2.imwrite(file_name, res)

    def choose_img(self):
        gray_img_list = os.listdir(self.base_dir)
        for i, img in enumerate(sorted(gray_img_list)):
            print(f"{i}: {img}")

        while True:
            try:
                img_to_colorize = int(input("Choose number of image to colorize: "))
                file_name = sorted(gray_img_list)[img_to_colorize]
                break
            except ValueError:
                print("Please choose a number.")

            except IndexError:
                print("Please choose a number from a list above.")


        if file_name:
            return file_name

    def __call__(self):
        img_to_colorize = self.choose_img()
        h, w, l_channel = self.handle_img(img_to_colorize)

        start_time = time.time()

        res = self.model.predict(l_channel)
        end_time = time.time()

        res_time = end_time - start_time
        print(f"Model took: {res_time:.4f} seconds for prediction")
        self.save_img(res, h, w, img_to_colorize)

base_dir = "/Users/maxkucher/historical_colorizer /grayscaled_images"
dir_to_save = "colorized_images"

colorizer = Colorizer(model, base_dir, dir_to_save)
colorizer()