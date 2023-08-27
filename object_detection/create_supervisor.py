import argparse
from .rotate_affine import get_rotated_img_dict
from . import detect_contour
from .utils import get_dice_size
from . import create_ng_image
import os
from .utils import chippitch
from .config_chip import RATIO
import numpy as np
import cv2
import glob
import jpeg4py


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--basepath", default="/mnt/b238g-000/20230216.不良サンプル追加/D0RM-APA01-22/20230216111902_231K03060S")
    parser.add_argument("--background_path", default="./background")
    parser.add_argument("-x", "--ext", default=".jpg")
    parser.add_argument("-t", "--temp_save_dir", default='/home/localuser/Documents/Dataset/original/')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mtype", type=str, default='D0RM-APA01-22')
    parser.add_argument("-s", "--save_dir", type=str, default='/home/localuser/Documents/Dataset/artificial5')
    return parser.parse_args()


def create_supervisor(args):
    temp_save_dir = os.path.join(args.temp_save_dir, args.mtype, '0001_0_0')
    os.makedirs(temp_save_dir, exist_ok=True)
    chip_width, chip_height = get_dice_size.get_dice_size(args.mtype)
    if '82' in args.mtype:
        chip_width = (chip_height / 2 * np.sqrt(3))
    chip_pitch_info = chippitch.get_chippitch(args.mtype)
    chip_pitch_x: float = (chip_pitch_info[0][8] - chip_width) * RATIO
    chip_pitch_y: float = (chip_pitch_info[0][9] - chip_height) * RATIO
    background_width = int((chip_pitch_x * 2 + chip_width * RATIO) // 2 * 2) - 30
    background_height = int((chip_pitch_y * 2 + chip_height * RATIO) // 2 * 2) - 30
    dict_image_array = get_rotated_img_dict(args.basepath, args.ext, temp_save_dir)
    """dict_image_array = {}
    for image in list_image:
        dict_image_array[image] = jpeg4py.JPEG(image).decode()"""

    save_dir = os.path.join(args.save_dir, args.mtype, '0001_0_0')
    os.makedirs(save_dir, exist_ok=True)
    empty_images = detect_contour.get_contours(dict_image_array, chip_width, chip_height, args.mtype, temp_save_dir)
    empty_image = empty_images[0]
    empty_img = dict_image_array[empty_image]
    h, w = empty_img.shape[:2]
    y1 = h // 2 - background_height // 2
    y2 = y1 + background_height
    x1 = w // 2 - background_width // 2
    x2 = x1 + background_width
    bg = empty_img[y1:y2, x1:x2, :]
    save_dir = os.path.join(args.save_dir, args.mtype, '0001_0_0')
    os.makedirs(save_dir, exist_ok=True)
    create_ng_image.create(dict_image_array, save_dir, bg, args.mtype, chip_width, chip_height)


def main():
    args = get_args()
    create_supervisor(args)


if __name__ == "__main__":
    main()
