import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import tqdm
from kornia.enhance import Normalize
from kornia.geometry import transform as K
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import nms

from . import configs
from .datasets import pred_dataset
from .retinanet.retinanet_model import get_model

sys.modules['configs'] = configs
font = ImageFont.truetype("ipag.ttf", 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/mnt/hqnas-ai-img2-data/B238_NgImage/20230612_暫定処置/20230710170216_236G06580S')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='/home/f1004/Documents/TechnoPro/object_detection/models/efficientnetv2_m/train_202309302045/model.pt')
    parser.add_argument('--arg_file', '-a', type=str, default='/home/f1004/Documents/TechnoPro/object_detection/models/efficientnetv2_m/train_202309302045/args.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./Image')
    parser.add_argument('--img_save', action='store_true')
    args = parser.parse_args()
    return args


def get_coordinate(paths):
    seqs, lotnos, ts, rs = [], [], [], []
    for path in paths:
        items = os.path.splitext(os.path.basename(path))[0].split('_')
        seqs.append(int(items[0]))
        lotnos.append(items[1])
        ts.append(float(items[3]) * (-1))
        rs.append(float(items[2]) * 1000)
    return seqs, lotnos, torch.tensor(ts), torch.tensor(rs)


def get_region(image, height, width):
    items = os.path.splitext(os.path.basename(image))[0].split('_')
    t = float(items[3]) % 360 * (-1)
    rad = np.radians(t)
    theta = np.arctan(height / width)
    length = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    phi1 = rad - theta
    phi2 = np.pi - rad - theta
    center = np.array([length, length])
    lt = center + np.array([-np.cos(phi1), np.sin(phi1)]) * length
    rt = center + np.array([np.cos(phi2), np.sin(phi2)]) * length
    rb = center + np.array([np.cos(phi1), -np.sin(phi1)]) * length
    lb = center + np.array([-np.cos(phi2), -np.sin(phi2)]) * length
    region = np.array([lt, rt, rb, lb])
    return region


def collate_fn(batch):
    return tuple(zip(*batch))


def expand_box(box, image_width, image_height, chip_pitch_x, chip_pitch_y, margin_x, margin_y):
    box[0] = max(0, box[0] - chip_pitch_x / 2 - margin_x)
    box[1] = max(0, box[1] - chip_pitch_y / 2 - margin_y)
    box[2] = min(image_width - 1, box[2] + chip_pitch_x / 2 + margin_x)
    box[3] = min(image_height - 1, box[3] + chip_pitch_y / 2 + margin_y)
    return box


def predict(args, arguments):
    classes = arguments.classes
    classes.insert(0, "__background__")
    colors = arguments.colors
    model = get_model(len(classes), arch=arguments.arch, backbone=arguments.backbone, is_torch=arguments.is_torch, out_channels=arguments.out_channels, baseline=arguments.baseline, num_layers=arguments.num_layers, nms_thresh=0.3, score_thresh=0.5,
                      topk_candidates=arguments.topk_candidates, detections_per_img=arguments.detections_per_img, layers=arguments.returned_layers)
    params = torch.load(args.model_path, map_location=args.device)
    weights = params['weights']
    model.to(device=args.device)
    model.load_state_dict(weights)
    model.eval()
    height = params['height']
    width = params['width']
    save_dir = f'{args.save_dir}/{os.path.basename(args.image_dir)}'
    os.makedirs(save_dir, exist_ok=True)
    kornia_transform = nn.Sequential(
        K.Resize((height, width)),
    )
    dataset = pred_dataset.PredDataset(args.image_dir, args.ext)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    for batch in tqdm.tqdm(dataloader):
        images, paths = batch
        images = torch.stack(images)
        transformed_tensors = kornia_transform(images)
        with torch.no_grad():
            predictions = model(transformed_tensors)
        images = images.mul(255).add_(0.5).clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        for i, prediction in enumerate(predictions):
            path = paths[i]
            img = Image.fromarray(images[i])
            w, h = img.size
            boxes = prediction['boxes']
            scores = prediction['scores']
            labels = prediction['labels']
            ratio = torch.tensor([w / width, h / height] * 2, device=args.device)
            boxes.mul_(ratio)
            nms_indices = nms(boxes, scores, 0.3)
            boxes = boxes.to(torch.int32)
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]
            labels = labels[nms_indices]
            boxes = boxes.cpu().numpy()
            img_draw = ImageDraw.Draw(img)
            for j, box in enumerate(boxes):
                label = labels[j]
                score = scores[j]
                color = colors[label]
                label = classes[label]
                txt = '{}{}'.format(label, np.round(score, 2))
                text_box = img_draw.textbbox((x1, y1), txt, font=font, anchor='lb')
                img_draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=10)
                img_draw.rectangle(text_box, fill=color)
                img_draw.text(text_box[:2], txt, (0, 0, 0), font=font)


def main():

    args = parse_args()

    arg_file = args.arg_file
    with open(arg_file, 'rb') as f:
        arguments = pickle.load(f)
    start = time.perf_counter()
    predict(args, arguments)
    end = time.perf_counter()
    print(end - start)


if __name__ == "__main__":
    main()
