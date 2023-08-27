import cv2
import numpy as np
import tqdm
import jpeg4py
import os
from PIL import Image
import glob
from .utils import get_lot_info
from . import create_annotaion_xml
from box_utils import box_utils
from inpoly import inpoly2
from . import config_chip
import matplotlib.pyplot as plt


def get_region(image):
    height = 7000
    width = 9344
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


def get_contour(img, imgfile, chip_width, chip_height, mtype, save_dir, lower, upper):

    if np.max(img) < 100:
        return
    h, w = img.shape[:2]
    list_rectangle = np.empty((0, 4), dtype=np.int32)
    if mtype == 'D0RM-24A01-05':
        alpha = 6
        beta = 150
        gamma = 4
    else:
        alpha = 5
        beta = 100
        gamma = 4
    is_hex = ('82' in mtype)
    bw = box_utils.get_bw_image(img, 0, 'OTSU', alpha, beta, gamma)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = get_region(imgfile)
    for c in contours:
        if (lower < cv2.contourArea(c) < upper):
            if not is_hex:
                angle, center = box_utils.find_angle2(c, chip_width, chip_height)
                vertices = box_utils.get_vertices_sv(center, angle, chip_width, chip_height, is_hex, config_chip.RATIO)

            else:
                angle, center = box_utils.find_angle_hex2(c, chip_width, chip_height)
                vertices = box_utils.get_vertices_sv(center, angle, chip_width, chip_height, is_hex, config_chip.RATIO)
            if is_inpoly(region, vertices):
                center = (center[0], center[1])
                xmin = np.min(vertices[:, 0])
                ymin = np.min(vertices[:, 1])
                xmax = np.max(vertices[:, 0])
                ymax = np.max(vertices[:, 1])
                xmin = max(0, xmin - 20)
                ymin = max(0, ymin - 20)
                xmax = min(w - 1, xmax + 20)
                ymax = min(h - 1, ymax + 20)
                box = np.array([xmin, ymin, xmax, ymax]).astype(np.int32)
                list_rectangle = np.vstack((list_rectangle, box))
    create_annotaion_xml.create_xml(imgfile, list_rectangle, save_dir, w, h)
    pil_img = Image.fromarray(img)
    pil_img.save(f'{save_dir}/{os.path.basename(imgfile)}')


def get_contours(list_image, chip_width, chip_height, mtype, save_dir):
    RATIO = config_chip.RATIO

    p_bar = tqdm.tqdm(list_image)
    p_bar.set_description('detect regions of chips')
    LOWER_RATIO = 0.7
    UPPER_RATIO = 1.2
    is_hex = '82' in mtype
    if not is_hex:
        area = chip_width * chip_height
    else:
        area = chip_height / 2 * chip_height / 2 * np.sqrt(3) / 2 * 3
    lower = area * RATIO ** 2 * LOWER_RATIO
    upper = area * RATIO ** 2 * UPPER_RATIO
    empty_images = []

    for imgfile in p_bar:
        if isinstance(list_image, dict):
            img = list_image[imgfile]
        elif isinstance(list_image, list):
            img = jpeg4py.JPEG(imgfile).decode()
        if np.max(img) < 100:
            if isinstance(list_image, dict):
                list_image[imgfile] = None
            continue
        h, w = img.shape[:2]
        list_rectangle = np.empty((0, 4), dtype=np.int32)
        list_class = []
        if mtype == 'D0RM-24A01-05':
            alpha = 6
            beta = 150
            gamma = 4
        else:
            alpha = 5
            beta = 100
            gamma = 4

        bw = box_utils.get_bw_image(img, 0, 'OTSU', alpha, beta, gamma)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region = get_region(imgfile)
        for c in contours:
            _, size, _ = cv2.minAreaRect(c)
            if (lower < size[0] * size[1] < upper):
                if not is_hex:
                    angle, center = box_utils.find_angle2(c, chip_width, chip_height)
                    vertices = box_utils.get_vertices_sv(center, angle, chip_width, chip_height, is_hex, config_chip.RATIO)
                else:
                    angle, center = box_utils.find_angle_hex2(c, chip_width, chip_height)
                    vertices = box_utils.get_vertices_sv(center, angle, chip_width, chip_height, is_hex, config_chip.RATIO)
                if is_inpoly(region, vertices):
                    center = (center[0], center[1])
                    xmin = np.min(vertices[:, 0])
                    ymin = np.min(vertices[:, 1])
                    xmax = np.max(vertices[:, 0])
                    ymax = np.max(vertices[:, 1])
                    xmin = max(0, xmin - 5)
                    ymin = max(0, ymin - 5)
                    xmax = min(w - 1, xmax + 5)
                    ymax = min(h - 1, ymax + 5)
                    box = np.array([xmin, ymin, xmax, ymax]).astype(np.int32)
                    list_class.append('chip')
                    list_rectangle = np.vstack((list_rectangle, box))
        if len(list_rectangle) == 0:
            empty_images.append(imgfile)
        create_annotaion_xml.create_xml(imgfile, list_rectangle, save_dir, w, h, list_class)
    return empty_images


def execute(base_dir):
    list_dir = os.listdir(base_dir)
    for mtype in tqdm.tqdm(list_dir):
        path = os.path.join(base_dir, mtype)
        list_image = glob.glob(f'{path}/0001_0_0/*.jpg')
        chip_width, chip_height = get_lot_info.get_chipsize(mtype)
        get_contours(list_image, chip_width, chip_height, mtype, f'{path}/0001_0_0')


def main():
    base_dir = '/home/localuser/Documents/Rotate/ArtificialImage'
    list_dir = os.listdir(base_dir)
    for mtype in tqdm.tqdm(list_dir):
        if mtype != '4HB-HE70NC':
            continue
        path = os.path.join(base_dir, mtype, '0001_0_0')
        list_image = glob.glob(f'{path}/*.jpg')
        get_contours(list_image, 1000, 1000, mtype, path)


if __name__ == "__main__":
    main()
