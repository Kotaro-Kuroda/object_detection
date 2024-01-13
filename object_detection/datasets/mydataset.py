import glob
import json
import os
import xml.etree.ElementTree as ET

import cv2
import jpeg4py
import torch
import tqdm
from PIL import Image
from torchvision import transforms


class Json2List:
    def __init__(self, classes, height, width, resized_height, resized_width):
        self.classes = classes
        self.height = height
        self.width = width
        self.resized_height = resized_height
        self.resized_width = resized_width

    def __call__(self, json_path):
        boxes = []
        labels = []
        with open(json_path, 'r') as f:
            anno = json.load(f)
        for label in anno['labels']:
            bbox = label['bbox']
            c = label['label']
            x1, y1, x2, y2 = bbox
            x1 = x1 * self.resized_width / self.width
            x2 = x2 * self.resized_width / self.width
            y1 = y1 * self.resized_height / self.height
            y2 = y2 * self.resized_height / self.height
            boxes.append([x1, y1, x2, y2])
            index = self.classes.index(c)
            labels.append(index)
        anno = {'bboxes': boxes, 'labels': labels}
        return anno


class xml2list(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path):
        boxes = []
        labels = []
        num_objs = 0
        if os.path.exists(xml_path):
            xml = ET.parse(xml_path).getroot()
            for obj in xml.iter('object'):
                label = obj.find('name').text.strip()
                if label in self.classes:
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.classes.index(label))
                    num_objs += 1

        anno = {'bboxes': boxes, 'labels': labels}

        return anno, num_objs


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, train_dir, classes, ext='.jpg'):

        super().__init__()
        self.train_dir = train_dir
        self.classes = classes
        self.ext = ext
        self.image_list = self._get_image_list()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.valid_image_list, self.list_annotaion = self._get_valid_image_list()

    def _get_image_list(self):
        list_image = glob.glob(f'{self.train_dir}/*{self.ext}')
        return list_image

    def _get_valid_image_list(self):
        images = []
        annotations = []
        for image in tqdm.tqdm(self.image_list):
            xml_path = f'{os.path.splitext(image)[0]}.xml'
            # transform_anno = Json2List(self.classes, h, w, self.height, self.width)
            transform_anno = xml2list(self.classes)
            annotation, num_obj = transform_anno(xml_path)
            if num_obj > 0:
                images.append(self._preproc(image))
                annotations.append(annotation)

        return images, annotations

    def _get_size(self, image):
        img = Image.open(image)
        w, h = img.size
        return h, w

    def _preproc(self, image):
        if self.ext == '.jpg':
            image = jpeg4py.JPEG(image).decode()
        else:
            image = cv2.imread(image)
        h, w = image.shape[:2]
        return self.transform(image)

    def __getitem__(self, index):
        image = self.valid_image_list[index]
        annotations = self.list_annotaion[index]
        boxes = torch.as_tensor(annotations['bboxes'], dtype=torch.int64)
        labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
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
