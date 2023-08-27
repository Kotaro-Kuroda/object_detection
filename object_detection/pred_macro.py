import torch
import argparse
import numpy as np
import copy
from PIL import ImageDraw, ImageFont, Image
import time
import os
import tqdm
from kornia.geometry import transform as K
from torch import nn
from kornia.enhance import Normalize
from torchvision.ops import nms
import pickle
import cv2
import sys
from inpoly import inpoly2
from .retinanet.retinanet_model import get_model
from . import configs
sys.modules['configs'] = configs
font = ImageFont.truetype("ipag.ttf", 20)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/home/localuser/Documents/Dataset/macro/231J04J70S.tif')
    parser.add_argument('--model_path', type=str, default='/home/localuser/Documents/object_detection/models/efficientnetv2_m/soshidachi_202306151419/model.pt')
    parser.add_argument('--arg_file', '-a', type=str, default='/home/localuser/Documents/object_detection/models/efficientnetv2_m/soshidachi_202306151419/args.pkl')
    parser.add_argument('--patch_width', type=int, default=1024)
    parser.add_argument('--patch_height', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./Image2')
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


def is_inpoly(region, vertices):
    if vertices.shape == (4,):
        x1, y1, x2, y2 = vertices
        vertices = np.array([[x1, y1],
                             [x2, y1],
                             [x2, y2],
                             [x1, y2]])
    isin, _ = inpoly2(vertices, region)
    return np.all(isin)


def collate_fn(batch):
    return tuple(zip(*batch))


def expand_box(box, image_width, image_height, chip_pitch_x, chip_pitch_y, margin_x, margin_y):
    box[0] = max(0, box[0] - chip_pitch_x / 2 - margin_x)
    box[1] = max(0, box[1] - chip_pitch_y / 2 - margin_y)
    box[2] = min(image_width - 1, box[2] + chip_pitch_x / 2 + margin_x)
    box[3] = min(image_height - 1, box[3] + chip_pitch_y / 2 + margin_y)
    return box


def predict_rotate(args, arguments):
    classes = arguments.classes
    colors = arguments.colors
    classes.insert(0, "__background__")
    is_torch = arguments.get('is_torch')
    if is_torch is None:
        is_torch = False
    model = get_model(len(classes), arch=arguments.arch, backbone=arguments.backbone, is_torch=is_torch,
                      out_channels=arguments.out_channels, baseline=arguments.baseline,
                      num_layers=arguments.num_layers, nms_thresh=0.3, score_thresh=0.3,
                      topk_candidates=arguments.topk_candidates, detections_per_img=arguments.detections_per_img,
                      layers=arguments.returned_layers)
    params = torch.load(args.model_path, map_location=args.device)
    weights = params['weights']
    model.to(device=args.device)
    model.load_state_dict(weights)
    model.eval()
    mean = params['mean']
    std = params['std']
    height = params['height']
    width = params['width']
    kornia_transform = nn.Sequential(
        K.Resize((height, width)),
        Normalize(mean=mean, std=std)
    )
    save_dir = f'{args.save_dir}/{os.path.splitext(os.path.basename(args.image_path))[0]}'
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(args.image_path)
    images = []
    h, w = img.shape[:2]
    patch_height = args.patch_height
    patch_width = args.patch_width
    centers = []
    list_boxes = np.empty((0, 4), dtype=np.int32)
    list_scores = np.empty(0, dtype=np.float32)
    list_labels = np.empty(0, dtype=np.int64)
    for i in range(0, h - patch_height // 2, patch_height // 2):
        for j in range(0, w - patch_width // 2, patch_width // 2):
            im = img[i:i + patch_height, j:j + patch_width]
            images.append(im)
            cx = (j + j + patch_width) // 2
            cy = (i + i + patch_height) // 2
            centers.append((cx, cy))
    for i, image in enumerate(tqdm.tqdm(images)):
        center = centers[i]
        tensor = torch.from_numpy(image)
        tensor = tensor.to(args.device)
        tensor = tensor.permute(2, 0, 1).div(255).unsqueeze(0)
        transformed_tensor = kornia_transform(tensor)
        with torch.no_grad():
            predictions = model(transformed_tensor)
        for prediction in predictions:
            boxes = prediction['boxes']
            scores = prediction['scores']
            labels = prediction['labels']
            nms_indices = nms(boxes, scores, 0.5)
            boxes = boxes.to(torch.int32)
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]
            labels = labels[nms_indices]
            boxes = boxes.cpu().numpy()
            boxes[:, 0] += center[0] - patch_width // 2
            boxes[:, 1] += center[1] - patch_height // 2
            boxes[:, 2] += center[0] - patch_width // 2
            boxes[:, 3] += center[1] - patch_height // 2
            list_boxes = np.append(list_boxes, boxes, axis=0)
            scores = scores.cpu().numpy()
            list_scores = np.append(list_scores, scores)
            labels = labels.cpu().numpy()
            list_labels = np.append(list_labels, labels)
    nms_indices = nms(torch.from_numpy(list_boxes).to(torch.float32), torch.from_numpy(list_scores), 0.5)
    nms_indices = nms_indices.cpu().numpy()
    list_boxes = list_boxes[nms_indices]
    list_scores = list_scores[nms_indices]
    list_labels = list_labels[nms_indices]
    pil_image = Image.fromarray(img)
    img_draw = ImageDraw.Draw(pil_image)
    num = 0
    center = (w // 2, h // 2)

    for j, box in enumerate(list_boxes):
        distance = get_max_distance(center, box)
        x1, y1, x2, y2 = box
        score = list_scores[j]
        label = list_labels[j]
        color = colors[label]
        label = classes[label]
        if label != 'soshidachi':
            continue
        num += 1
        txt = '{}{}'.format(label, np.round(score, 2))
        text_box = img_draw.textbbox((x1, y1), txt, font=font, anchor='lb')
        img_draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
        # img_draw.rectangle(text_box, fill=color)
        # img_draw.text(text_box[:2], txt, (0, 0, 0), font=font)
    pil_image.save(os.path.join(save_dir, os.path.splitext(os.path.basename(args.image_path))[0] + '.jpg'), quality=95)
    print(num)


def get_max_distance(center, vertices):
    if (vertices.ndim == 1 and len(vertices) == 4):
        vertices = np.array([[vertices[0], vertices[1]], [vertices[2], vertices[1]], [vertices[2], vertices[3]], [vertices[0], vertices[3]]])

    if (vertices.ndim != 2 or vertices.shape[1] != 2):
        raise ValueError('verticesは n x 2 の行列か 1 x 4のベクトルを指定してください')
    center = np.array(center)
    return np.max(np.linalg.norm(vertices - center, axis=1))


def main():

    args = parse_args()
    arg_file = args.arg_file
    with open(arg_file, 'rb') as f:
        arguments = pickle.load(f)
    start = time.perf_counter()
    predict_rotate(args, arguments)
    end = time.perf_counter()
    print(end - start)


if __name__ == "__main__":
    main()
