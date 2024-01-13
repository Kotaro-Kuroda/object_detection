import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.ops import nms

from . import configs
from .datasets import pred_dataset
from .retinanet.retinanet_model import get_model

sys.modules['configs'] = configs
font = ImageFont.truetype("ipag.ttf", 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='/home/f1004/Documents/TechnoPro/object_detection/models/efficientnetv2_m/train_202309302045/model.pt')
    parser.add_argument('--arg_file', '-a', type=str, default='/home/f1004/Documents/TechnoPro/object_detection/models/efficientnetv2_m/train_202309302045/args.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./Image')
    parser.add_argument('--img_save', action='store_true')
    args = parser.parse_args()
    return args


def collate_fn(batch):
    return tuple(zip(*batch))


def predict(args, cfg):
    classes = cfg.classes
    classes.insert(0, "__background__")
    colors = cfg.colors
    model = get_model(len(classes), arch=cfg.arch, backbone=cfg.backbone, is_torch=cfg.is_torch, out_channels=cfg.out_channels, baseline=cfg.baseline, num_layers=cfg.num_layers, nms_thresh=0.3, score_thresh=0.5,
                      topk_candidates=cfg.topk_candidates, detections_per_img=cfg.detections_per_img, layers=cfg.returned_layers)
    params = torch.load(args.model_path, map_location=args.device)
    weights = params['weights']
    model.to(device=args.device)
    model.load_state_dict(weights)
    model.eval()
    height = params['height']
    width = params['width']
    save_dir = f'{args.save_dir}/{os.path.basename(args.image_dir)}'
    os.makedirs(save_dir, exist_ok=True)

    dataset = pred_dataset.PredDataset(args.image_dir, args.ext)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    for batch in tqdm.tqdm(dataloader):
        images, paths = batch
        with torch.no_grad():
            predictions = model(images)
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
                x1, y1, x2, y2 = box
                label = labels[j]
                score = scores[j]
                color = colors[label]
                label = classes[label]
                txt = '{}{}'.format(label, np.round(score, 2))
                text_box = img_draw.textbbox((x1, y1), txt, font=font, anchor='lb')
                img_draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=10)
                img_draw.rectangle(text_box, fill=color)
                img_draw.text(text_box[:2], txt, (0, 0, 0), font=font)
            img.save(os.path.join(save_dir, os.path.basename(path)))


def main():

    args = parse_args()

    arg_file = args.arg_file
    with open(arg_file, 'rb') as f:
        cfg = pickle.load(f)
    start = time.perf_counter()
    predict(args, cfg)
    end = time.perf_counter()
    print(end - start)


if __name__ == "__main__":
    main()
