import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import jpeg4py


class SuperVisorDataset(Dataset):
    def __init__(self, dataset_path, ext):
        self.dataset_path = dataset_path
        self.x = self.load_dataset_folder(ext)
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        path = self.x[idx]
        img = jpeg4py.JPEG(path).decode()
        return path, self.transform(img)

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, ext):
        return sorted(list(glob.glob(os.path.join(self.dataset_path, f"*{ext}"))))
