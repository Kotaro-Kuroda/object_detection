import glob
import os

import jpeg4py
from torch.utils.data import Dataset
from torchvision import transforms


class PredDataset(Dataset):
    def __init__(self, image_dir, ext):
        super().__init__()
        self.image_dir = image_dir
        self.list_image = glob.glob(f'{image_dir}/*{ext}')
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        path = self.list_image[index]
        img = jpeg4py.JPEG(path).decode()
        tensor_img = self.transform(img)
        return tensor_img, os.path.basename(path)

    def __len__(self):
        return len(self.list_image)
