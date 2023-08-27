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
import glob
import pickle
import sys
from inpoly import inpoly2
from box_utils import box_utils, rotate, save_rotated_image
from torch.utils.data import DataLoader
from .datasets import pred_dataset
from .retinanet.retinanet_model import get_model
from .utils import get_dice_size
from .rotate_affine import rotate_affine
from . import configs
from .utils import chippitch
from .config_chip import RATIO
sys.modules['configs'] = configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/mnt/hqnas-ai-img2-data/B238_NgImage/20230612_暫定処置/20230710170216_236G06580S')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='/home/localuser/Documents/B238/frcnn/retinanet/all/efficientnetv2/model_230602.pt')
    parser.add_argument('--arg_file', '-a', type=str, default='/home/localuser/Documents/B238/frcnn/retinanet/all/efficientnetv2/args_230602.pkl')
    parser.add_argument('--chip_img_save_dir', type=str, default='/home/localuser/Documents/waferlot/Image')
    parser.add_argument('--mtype', type=str, default='D0RS-70A03-10')
    parser.add_argument('--margin_x', type=int, default=40)
    parser.add_argument('--margin_y', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./Image2')
    parser.add_argument('--cut', action='store_true')
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
    classes.insert(0, "__background__")
    colors = arguments.colors
    model = get_model(len(classes), arch=arguments.arch, backbone=arguments.backbone, is_torch=arguments.is_torch, out_channels=arguments.out_channels, baseline=arguments.baseline, num_layers=arguments.num_layers, nms_thresh=0.3, score_thresh=0.5,
                      topk_candidates=arguments.topk_candidates, detections_per_img=arguments.detections_per_img, layers=arguments.returned_layers)
    params = torch.load(args.model_path, map_location=args.device)
    weights = params['weights']
    model.to(device=args.device)
    model.load_state_dict(weights)
    model.eval()
    mean = params['mean']
    std = params['std']
    height = params['height']
    width = params['width']
    save_dir = f'{args.save_dir}/{os.path.basename(args.image_dir)}'
    os.makedirs(save_dir, exist_ok=True)
    kornia_transform = nn.Sequential(
        K.Resize((height, width)),
        Normalize(mean=mean, std=std)
    )
    list_path = glob.glob(os.path.join(args.image_dir, f'*{args.ext}'))
    dict_coord = {}
    for path in list_path:
        items = os.path.splitext(os.path.basename(path))[0].split('_')
        radius, theta = items[-2:]
        dict_coord[(np.float32(radius), np.float32(theta))] = path
    dataset = pred_dataset.PredDataset(args.image_dir, args.ext)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    mtype = args.mtype
    chip_width, chip_height = get_dice_size.get_dice_size(mtype)
    is_hex = '82' in mtype
    margin_x = args.margin_x
    margin_y = args.margin_y
    os.makedirs(args.chip_img_save_dir, exist_ok=True)
    font = ImageFont.truetype("ipag.ttf", 50)
    np.random.seed(0)
    chip_pitch_info = chippitch.get_chippitch(mtype)
    chip_pitch_x: float = (chip_pitch_info[0][8] - chip_width) * RATIO
    chip_pitch_y: float = (chip_pitch_info[0][9] - chip_height) * RATIO
    for batch in tqdm.tqdm(dataloader):
        images, paths = batch
        images = torch.stack(images)
        _, _, ts, rs = get_coordinate(paths)
        ts = ts.to(device=args.device)
        *_, original_height, original_width = images[0].shape
        diagonal = int(torch.norm(torch.tensor([original_height, original_width]).float()).item())
        images = rotate_affine(images, ts, diagonal, args.device)
        transformed_tensors = kornia_transform(images)
        with torch.no_grad():
            predictions = model(transformed_tensors)
        images = images.mul(255).add_(0.5).clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        ts = ts.cpu()
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
            if args.img_save:
                img_draw = ImageDraw.Draw(img)
            region = get_region(path, original_height, original_width)
            for j, box in enumerate(boxes):
                if not is_inpoly(region, box):
                    continue
                x1, y1, x2, y2 = box
                if args.img_save:
                    score = scores[j].item()
                    label = labels[j].item()
                    color = colors[label]
                    label = classes[label]
                    txt = '{}{}'.format(label, np.round(score, 2))
                    text_box = img_draw.textbbox((x1, y1), txt, font=font, anchor='lb')
                    img_draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=10)
                    img_draw.rectangle(text_box, fill=color)
                    img_draw.text(text_box[:2], txt, (0, 0, 0), font=font)
                if args.cut:
                    expanded_box = expand_box(box, w, h, chip_pitch_x, chip_pitch_y, margin_x, margin_y)
                    x1, y1, x2, y2 = expanded_box
                    rect_image = images[i][y1:y2, x1:x2, :3]
                    if mtype == 'D0RM-AMA-01-22':
                        beta = 100
                    else:
                        beta = 150
                    angle, center = box_utils.find_angle(rect_image, chip_width, chip_height, mtype, is_hex, RATIO, alpha=5.0, beta=beta, gamma=4.0)
                    if angle is None or center is None:
                        continue
                    rot_img = rotate.rotate(rect_image, angle, center, chip_width, chip_height, margin_x, margin_y, RATIO)
                    save_path = os.path.join(args.chip_img_save_dir, f'{os.path.splitext(os.path.basename(path))[0]}_{str(j)}.jpg')
                    save_rotated_image.save_image(rot_img, save_path, quality=95)
            if args.img_save:
                img.save(os.path.join(save_dir, f'{os.path.splitext(os.path.basename(path))[0]}.jpg'))


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
