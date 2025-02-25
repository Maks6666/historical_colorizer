from os import listdir
import time
from skimage.color import rgb2lab
from models import emb_model_1, extractor_1, emb_model_2, extractor_2, extractor_3, emb_model_3
import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from datasets import custom_image_transformer
import cv2

from tools import choose_file


device = "mps" if torch.backends.mps.is_available() else "cpu"


class ColorizerApp:
    def __init__(self, image, dir_to_save, model_marker, extractor):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dir_to_save = dir_to_save
        self.model_marker = model_marker
        self.embedding_extractor = extractor
        self.model = self.load_model()
        self.image = image
        self.transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor()
        ])


    def load_model(self):
        if self.model_marker == "1":
            return emb_model_1()
        elif self.model_marker == "2":
            return emb_model_2()
        elif self.model_marker == "3":
            return emb_model_3()


    def output_result(self):
        idx = len(os.listdir(self.dir_to_save))+1
        os.makedirs(self.dir_to_save, exist_ok=True)


        start = time.time()

        if os.path.isfile(self.image):

            image = Image.open(self.image)
            width, height = image.size

            img, gray_img = custom_image_transformer(self.image)

            self.model.to(self.device)
            emd_model = self.embedding_extractor()
            emd_model.to(self.device)

            gray_img = gray_img.to(self.device)
            img = img.to(self.device)

            try:
                print("Embedding returned")
                embedding = emd_model(gray_img, return_emb=True)
            except RuntimeError and TypeError:
                embedding = emd_model(gray_img)

            res = self.model.predict(img, embedding)

            if len(res.shape) == 4:
                res = res.squeeze(0)

            if np.max(res) <= 1:
                res = (res * 255).astype(np.uint8)

            res = cv2.resize(res, (width, height))
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

            path = os.path.join(self.dir_to_save, f"{idx}.jpg")

            cv2.imwrite(path, res)

            end = time.time()

            print(f"Done! Proces took: {end-start:.2f}s")

            return res

root_dir = "grayscaled_images"
image = choose_file(root_dir)
dir_to_save = "colorized_images"


colorizer = ColorizerApp(image, dir_to_save, model_marker="3", extractor=extractor_3)
colorizer.output_result()
