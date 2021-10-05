import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

#solve the os.path issue
class DeepFashionDataset(Dataset):
    def __init__(self, txt_file, root, transform=None, landmarks=None, bbox=None):
        self.annotations = pd.read_csv(txt_file)
        self.root = root
        self.transform = transform
        self.landmarks = landmarks
        self.bbox = bbox

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = #os.path.join(self.root, self.annotations.iloc[index, :])
        image = io.imread(img_path)
        y_label = #torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)

