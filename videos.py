import cv2
import torch
from torchvision import transforms
from datasets import CustomVideoDataset
from torch.utils.data import DataLoader
from models import teacher_model, student_model, additional_model

import time
from tqdm import tqdm
from tools import choose_file
import random


class VideoCreator:
    def __init__(self, path, output_path, model_marker, device):
        self.path = path
        self.output_path = output_path

        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.cap = cv2.VideoCapture(self.path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.device = device
        self.model = None
        self.model_marker = model_marker
        self.frames = []

    def __call__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            upd_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preprocessed_frame = self.transformer(upd_frame)
            self.frames.append(preprocessed_frame)

        self.cap.release()
        video_data = CustomVideoDataset(self.frames)
        video_dl = DataLoader(video_data, batch_size=1, shuffle=False)

        if self.model_marker == "l":
            self.model = teacher_model()
        elif self.model_marker == "s":
            self.model = student_model()
        elif self.model_marker == "a":
            self.model = additional_model()

        if self.model:
            self.model.to(self.device)

        start = time.time()
        for batch in tqdm(video_dl):
            batch = batch.to(self.device)
            results = self.model.predict(batch)
            for res in results:
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
                res = cv2.resize(res, (self.width, self.height))
                self.out.write(res)

        finish = time.time()
        print(f"Done! Process took: {finish-start:.2f}s")
        self.out.release()

device = "mps" if torch.backends.mps.is_available() else "cpu"
num = random.randint(1, 1000)


path = choose_file("grayscaled_videos")
name = path.split("/")[-1]
output_path = f"colorized_videos/colorized_{name}"

colorizer = VideoCreator(path, output_path, "l", device)
colorizer()





