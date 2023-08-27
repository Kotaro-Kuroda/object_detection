import xml.etree.ElementTree as ET
from glob import glob
from torchvision import transforms
import os
import torch
import tqdm
import jpeg4py
from torchvision.transforms import functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import cv2


class xml2list(object):
    def __init__(self, classes, height, width):
        self.classes = classes
        self.height = height
        self.width = width

    def __call__(self, xml_path):
        boxes = []
        labels = []
        num_objs = 0
        if os.path.exists(xml_path):
            xml = ET.parse(xml_path).getroot()
            size = xml.find('size')
            height = int(size.find('height').text)
            width = int(size.find('width').text)
            for obj in xml.iter('object'):
                label = obj.find('name').text.strip()
                if label == 'チップ':
                    label = 'chip'
                if label in self.classes:
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text) * self.width / width
                    ymin = float(bndbox.find('ymin').text) * self.height / height
                    xmax = float(bndbox.find('xmax').text) * self.width / width
                    ymax = float(bndbox.find('ymax').text) * self.height / height
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.classes.index(label))
                    num_objs += 1

        anno = {'bboxes': boxes, 'labels': labels}

        return anno, num_objs


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, train_dir, height, width, classes, multi, base_dir, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, ext='jpg', p=1.0, mtype='all'):

        super().__init__()
        self.train_dir = train_dir
        self.base_dir = base_dir
        self.height = height
        self.width = width
        self.classes = classes
        self.multi = multi
        self.p = p
        self.ext = ext
        self.mtype = mtype
        self.image_list = self._get_image_list()
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.valid_image_list, self.list_annotaion, self.list_num_obje = self._get_valid_image_list()
        # self.mean = torch.mean(images_tensor, dim=(0, 2, 3))
        # self.std = torch.var(images_tensor, dim=(0, 2, 3), unbiased=True)

        # self.valid_image_list = [F.normalize(image, self.mean, self.std, inplace=True) for image in self.valid_image_list]

    # 画像をリストで取得

    def _get_image_list(self):
        if self.multi:
            list_dir = [dire for dire in os.listdir(
                self.train_dir) if os.path.isdir(os.path.join(self.train_dir, dire))]
        else:
            list_dir = [self.train_dir]
        list_img = []
        if self.mtype == 'all':
            for dire in list_dir:
                list_img += glob(os.path.join(self.train_dir, dire, '0001_0_0', f'*{self.ext}'))
        else:
            for dire in list_dir:
                list_img += glob(os.path.join(self.train_dir, dire, '0001_0_0', f'*{self.ext}'))
            if self.base_dir:
                list_dir = [dire for dire in os.listdir(
                    self.base_dir) if os.path.isdir(os.path.join(self.base_dir, dire))]
                for dire in list_dir:
                    list_img += glob(os.path.join(self.base_dir, dire, '0001_0_0', f'*{self.ext}'))
        return list_img

    # 取得した画像のうち、物体が写っているものだけ取得
    def _get_valid_image_list(self):
        list_image = []
        list_annotation = []
        list_obje_num = []
        for image in self.image_list:
            xml_path = f'{os.path.splitext(image)[0]}.xml'
            transform_anno = xml2list(self.classes, self.height, self.width)
            annotations, obje_num = transform_anno(xml_path)
            if obje_num > 0:
                list_image.append(image)
                list_annotation.append(annotations)
                list_obje_num.append(obje_num)
        indices = np.random.choice(range(len(list_image)), int(self.p * len(list_image)), replace=False)
        images = []
        annotations = []
        obje_nums = []
        p_bar = tqdm.tqdm(range(len(list_image)))
        p_bar.set_description('load images')
        for i in p_bar:
            if i in indices:
                images.append(self._preproc(list_image[i]))
                annotations.append(list_annotation[i])
                obje_nums.append(list_obje_num[i])
        return images, annotations, obje_nums

    def _preproc(self, image):
        if self.ext == '.jpg':
            image = jpeg4py.JPEG(image).decode()
        else:
            image = cv2.imread(image)
        return self.transform(image)

    def __getitem__(self, index):
        image = self.valid_image_list[index]
        annotations = self.list_annotaion[index]
        obje_num = self.list_num_obje[index]
        if obje_num == 0:
            boxes = torch.empty((0, 4), dtype=torch.int64)
            labels = torch.empty((0), dtype=torch.int64)
            area = torch.empty((0), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(annotations['bboxes'], dtype=torch.int64)
            labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((obje_num,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels + 1
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return image, target

    def __len__(self):
        return len(self.valid_image_list)


def dataloader(train_dir, dataset_class, batch_size, scale, multi):
    dataset = MyDataset(train_dir, scale, dataset_class, multi)
    torch.manual_seed(2020)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    return train_dataloader
