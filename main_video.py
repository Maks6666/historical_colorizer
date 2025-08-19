from gan_1 import model
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from video_dataset import VideoDataset
from tqdm import tqdm
import time
import os


class VideoColorizer:
    def __init__(self, model, base_dir, device):
        self.model = model
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        self.base_dir = base_dir
        self.device = device
        self.dir_to_save = "colorized_videos"
        os.makedirs(self.dir_to_save, exist_ok=True)
        self.frames = []

    def choose_video(self):
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
        return file_name

    def handle_frame(self, frame):
        upd_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed_frame = self.transformer(upd_frame)
        return transformed_frame

    def process_model_output(self, output):
        if output.max() <= 1.0:
            output = (output * 255).astype(np.uint8)
        else:
            output = output.astype(np.uint8)
        return output

    def __call__(self):
        video_to_colorize = self.choose_video()
        full_path = os.path.join(self.base_dir, video_to_colorize)
        # print(f"Processing video: {full_path}")

        cap = cv2.VideoCapture(full_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25.0

        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # base_name, _ext = os.path.splitext(video_to_colorize)
        output_path = os.path.join(self.dir_to_save, f"colorized_{video_to_colorize}")

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            upd_frame = self.handle_frame(frame)
            self.frames.append(upd_frame)
        cap.release()

        frames_data = VideoDataset(self.frames)
        frames_dl = DataLoader(frames_data, batch_size=1)

        start = time.time()
        for batch in tqdm(frames_dl):
            batch = batch.to(self.device)

            results = self.model.predict(batch)
            processed_frame = self.process_model_output(results)

            processed_frame = cv2.resize(processed_frame, (width, height))
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

            out.write(processed_frame)

        finish = time.time()
        print(f"Done! Process took: {finish - start:.2f}s")
        out.release()

base_dir = "/Users/maxkucher/historical_colorizer /grayscaled_videos"
device = "mps" if torch.backends.mps.is_available() else "cpu"

colorizer = VideoColorizer(model, base_dir, device)
colorizer()
