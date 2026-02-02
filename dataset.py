import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image Index"]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx]["Pneumonia"]

        if self.transform:
            image = self.transform(image)

        return image, label
