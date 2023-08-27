import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import tqdm
import argparse
import kornia
import torch
from box_utils import box_utils, rotate
from inpoly import inpoly2
import copy
from . import create_annotaion_xml
from .config_chip import RATIO


def get_region(image):
    height = 7000
    width = 9344
    items = os.path.splitext(os.path.basename(image))[0].split('_')
    r = float(items[2]) * 1000
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lotnumber', default='219H06A10S.20210924181813')
    args = parser.parse_args()
    return args


def rotate_image_kornia(bg, img, angle, device='cuda'):
    *_, bg_h, bg_w = bg.shape
    *_, c, h, w = img.shape
    if c == 3:
        img = kornia.color.rgb_to_rgba(img, 1.0)

    center = (w // 2, h // 2)
    center = torch.tensor([center]).to(device=device, dtype=torch.float32)
    a = np.radians(angle)
    w_rot = int(np.ceil(w * np.abs(np.cos(a)) + h * np.abs(np.sin(a))))
    h_rot = int(np.ceil(w * np.abs(np.sin(a)) + h * np.abs(np.cos(a))))
    scale = torch.ones((1, 2)).to(device)
    angle = torch.ones(1) * angle
    angle = angle.to(device)
    trans = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)

    trans[0][0][2] += -center[0][0] + w_rot / 2
    trans[0][1][2] += -center[0][1] + h_rot / 2
    rot_img = kornia.geometry.transform.warp_affine(img, trans, (h_rot, w_rot))
    alpha = rot_img[:, 3, :, :]
    alpha = kornia.color.grayscale_to_rgb(alpha)
    half_width = int(w_rot / 2)
    half_height = int(h_rot / 2)
    bg_center_x, bg_center_y = int(bg_w / 2), int(bg_h / 2)
    y1 = bg_center_y - half_height
    y2 = bg_center_y + half_height
    x1 = bg_center_x - half_width
    x2 = bg_center_x + half_width
    if y1 < 0 or x1 < 0:
        return rotate_image_kornia(bg, img, np.random.uniform(-180, 180), device)
    else:
        return bg


def _rotate_chip(img, angle):
    h, w, c = img.shape
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    center = int(w / 2), int(h / 2)
    trans = cv2.getRotationMatrix2D(center, angle, 1)

    a = np.radians(angle)
    w_rot = int(np.ceil(w * np.abs(np.cos(a)) + h * np.abs(np.sin(a))))
    h_rot = int(np.ceil(w * np.abs(np.sin(a)) + h * np.abs(np.cos(a))))
    trans[0][2] += -w / 2 + w_rot / 2
    trans[1][2] += -h / 2 + h_rot / 2
    dst = np.zeros((h_rot, w_rot, 4))
    rot_img = cv2.warpAffine(
        img, trans, (w_rot, h_rot), dst=dst)
    return rot_img


def rotate_image(bg, img, angle, gw, gh):
    bg_h, bg_w, _ = bg.shape
    rot_img = _rotate_chip(img, angle)
    h_rot, w_rot = rot_img.shape[:2]
    rot_img[:, :, 3] = np.where(rot_img[:, :, 3] > 0, 255, 0)
    alpha = rot_img[:, :, 3]
    alpha = cv2.cvtColor(
        alpha, cv2.COLOR_GRAY2BGR)  # grayをBGRに
    alpha = alpha / 255.0

    half_width = int(w_rot / 2)
    half_height = int(h_rot / 2)

    bg_center_x, bg_center_y = int(bg_w / 2), int(bg_h / 2)
    start_y = bg_center_y - half_height
    end_y = bg_center_y + half_height
    start_x = bg_center_x - half_width
    end_x = bg_center_x + half_width
    if start_y < 0 or start_x < 0 or end_y > gh - 1 or end_x > gw - 1:
        return rotate_image(bg, img, np.random.uniform(-180, 180), gw, gh)
    else:
        bg[start_y:end_y, start_x: end_x, :3] = bg[start_y: end_y, start_x: end_x, :3] * (1 - alpha[:half_height * 2, :half_width * 2, :3]).astype('uint8')
        bg[start_y:end_y, start_x: end_x, :3] = bg[start_y:end_y, start_x: end_x, :3] + rot_img[:half_height * 2, :half_width * 2, :3] * (alpha[:half_height * 2, :half_width * 2, :3]).astype('uint8')
        return bg, angle


def get_next_image_name(x, y, direction):
    name = ''
    if direction == 'left':
        x += 1
    elif direction == 'right':
        x -= 1
    elif direction == 'above':
        y -= 1
    elif direction == 'under':
        y += 1
    name = str(x).zfill(2) + '_' + str(y).zfill(2)
    return name


def image_composition(image, bg, loc, region, list_rectangle, box, base):
    h, w, _ = image.shape
    center_x = int((loc[0] + loc[2]) / 2)
    center_y = int((loc[1] + loc[3]) / 2)
    bg_h, bg_w, c = bg.shape

    y1 = center_y - bg_h // 2
    y2 = center_y + bg_h // 2
    x1 = center_x - bg_w // 2
    x2 = center_x + bg_w // 2
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        list_rectangle = np.vstack((list_rectangle, base))
        return image, list_rectangle
    if not is_inpoly(region, np.array([x1, y1, x2, y2])):
        list_rectangle = np.vstack((list_rectangle, base))
        return image, list_rectangle
    if c == 3:
        image = paste(image, bg, x1, x2, y1, y2)
    else:
        alpha = bg[:, :, 3]
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # grayをBGRに
        alpha = alpha / 255.0
        image = transparent(image, bg, alpha, x1, x2, y1, y2)
    list_rectangle = np.vstack((list_rectangle, box))
    return image, list_rectangle


def transparent(bg, img, alpha, x1, x2, y1, y2):
    bg[y1:y2, x1:x2, :3] = bg[y1:y2, x1:x2, :3] * (1 - alpha[:(y2 - y1), :(x2 - x1), :3]).astype('uint8')
    bg[y1:y2, x1:x2, :3] = bg[y1:y2, x1:x2, :3] + (img[:(y2 - y1), :(x2 - x1), :3] * alpha[:(y2 - y1), :(x2 - x1), :3]).astype('uint8')
    return bg


def paste(bg, img, x1, x2, y1, y2):
    bg[y1:y2, x1:x2, :3] = img[:(y2 - y1), :(x2 - x1), :3]
    return bg


def get_under_above_chip(list_coord, base, tmp_h):
    column = list_coord[np.maximum(list_coord[:, 0], base[0]) < np.minimum(list_coord[:, 2], base[2])]
    aboves = column[column[:, 3] < base[1]]
    unders = column[column[:, 1] > base[3]]
    if len(aboves) > 0:
        above = aboves[np.argmax(aboves[:, 3])]
    else:
        above = None
    if len(unders) > 0:
        under = unders[np.argmin(unders[:, 1])]
    else:
        under = None
    return above, under


def get_under_above_chips(list_coord, base, tmp_h):
    above, under = get_under_above_chip(list_coord, base, tmp_h)

    c1 = base[1] - above[3] < tmp_h if above is not None else False

    c2 = under[1] - base[3] < tmp_h if under is not None else False
    return (c1 or c2)


def get_left_right_chips(list_coord, base, tmp_w):
    row = list_coord[np.maximum(list_coord[:, 1], base[1]) < np.minimum(list_coord[:, 3], base[3])]
    lefts = row[row[:, 2] < base[0]]
    rights = row[row[:, 0] > base[2]]
    if len(lefts) > 0:
        left = lefts[np.argmax(lefts[:, 2])]
        c1 = base[0] - left[2] < tmp_w
    else:
        c1 = False
    if len(rights) > 0:
        right = rights[np.argmin(rights[:, 1])]
        c2 = right[0] - base[2]
    else:
        c2 = False
    return (c1 or c2)


def around_chips(list_coord, base, tmp_w, tmp_h):
    c1 = get_under_above_chips(list_coord, base, tmp_h)
    c2 = get_left_right_chips(list_coord, base, tmp_w)

    above, under = get_under_above_chip(list_coord, base, tmp_h)

    c3 = get_left_right_chips(list_coord, above, tmp_w) if above is not None else False
    c4 = get_left_right_chips(list_coord, under, tmp_w) if under is not None else False
    return c1 or c2 or c3 or c4


def rotate_vertices(vertices, center, angle):
    rad = np.deg2rad(angle)
    rotate_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    vertices = vertices - center
    rot_vertices = np.matmul(vertices, rotate_matrix)
    rot_vertices = rot_vertices + center
    return rot_vertices


def create(list_image, save_dir, background, mtype, chip_width, chip_height):
    pbar = tqdm.tqdm(list_image)
    pbar.set_description('create artificial abnormal images')
    is_hex = '82' in mtype
    bg_img = background
    for image in pbar:
        list_coord = np.empty((0, 4), dtype=np.int32)
        xml_path = f'{os.path.splitext(image)[0]}.xml'
        if not os.path.exists(xml_path):
            continue
        if isinstance(list_image, dict):
            img = list_image[image]
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(list_image, list):
            img = cv2.imread(image)
        h, w = img.shape[:2]
        xml = ET.parse(xml_path).getroot()
        list_objects = xml.findall('object')
        indices = np.random.choice(range(len(list_objects)), size=int(len(list_objects) * 0.05), replace=False)
        region = get_region(image)

        list_rectangle = np.empty((0, 4), dtype=np.int32)
        for i, obj in enumerate(list_objects):
            random = np.random.random()
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            base = np.array([xmin, ymin, xmax, ymax])
            if i not in indices:
                list_rectangle = np.vstack((list_rectangle, base))
                continue
            else:

                if not is_inpoly(region, base):
                    list_rectangle = np.vstack((list_rectangle, base))
                    continue
                random = np.random.random()
                rect_image = img[ymin:ymax, xmin:xmax, :]
                angle, center = box_utils.find_angle(rect_image, chip_width, chip_height, mtype, is_hex, RATIO)
                center_abs = (center[0] + xmin, center[1] + ymin)
                vertices = box_utils.get_vertices_sv(center_abs, angle, chip_width, chip_height, is_hex, RATIO)
                tmp = rotate.rotate2(rect_image, angle, center, chip_width, chip_height, is_hex, RATIO)
                tmp_h, tmp_w, tmp_c = tmp.shape
                c1 = get_under_above_chips(list_coord, base, tmp_h)
                c2 = get_left_right_chips(list_coord, base, tmp_w)
                if c1 or c2:
                    list_rectangle = np.vstack((list_rectangle, base))
                    continue
                list_coord = np.vstack((list_coord, base))
                bg = copy.deepcopy(bg_img)
                if tmp_c > 3:
                    alpha = tmp[:, :, 3]
                    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # grayをBGRに
                    alpha = alpha / 255.0
                bg_h, bg_w, _ = bg.shape
                bg_center_x, bg_center_y = bg_w // 2, bg_h // 2
                loc = [xmin, ymin, xmax, ymax]
                if random < 0.3:
                    """
                    θズレ
                    """
                    rotate_angle = np.random.uniform(-180, 180)
                    bg, rotate_angle = rotate_image(bg, tmp, rotate_angle, w, h)
                    rot_vertices = rotate_vertices(vertices, center_abs, rotate_angle - angle)
                    box = np.array([np.min(rot_vertices[:, 0]) - 20, np.min(rot_vertices[:, 1]) - 20, np.max(rot_vertices[:, 0]) + 20, np.max(rot_vertices[:, 1]) + 20], dtype=np.int32)

                elif 0.3 <= random < 0.6:
                    """
                    yズレ
                    """
                    y1 = 0
                    y2 = tmp_h
                    x1 = bg_center_x - tmp_w // 2
                    x2 = bg_center_x + tmp_w // 2
                    if tmp_c > 3:
                        bg = transparent(bg, tmp, alpha, x1, x2, y1, y2)
                    else:
                        bg = paste(bg, tmp, x1, x2, y1, y2)

                    center_x = int((loc[0] + loc[2]) / 2)
                    center_y = int((loc[1] + loc[3]) / 2)
                    y1 = center_y - bg_h // 2
                    y2 = y1 + tmp_h
                    x1 = center_x - tmp_w // 2
                    x2 = center_x + tmp_w // 2
                    box = np.array([x1 - 20, y1 - 20, x2 + 20, y2 + 20], dtype=np.int32)
                elif 0.6 <= random < 0.9:
                    """
                    xズレ
                    """
                    x1 = 0
                    x2 = tmp_w
                    y1 = bg_center_y - tmp_h // 2
                    y2 = bg_center_y + tmp_h // 2
                    if tmp_c > 3:
                        bg = transparent(bg, tmp, alpha, x1, x2, y1, y2)
                    else:
                        bg = paste(bg, tmp, x1, x2, y1, y2)
                    center_x = int((loc[0] + loc[2]) / 2)
                    center_y = int((loc[1] + loc[3]) / 2)
                    y1 = center_y - tmp_h // 2
                    y2 = y1 + tmp_h
                    x1 = center_x - bg_w // 2
                    x2 = x1 + tmp_w
                    box = np.array([x1 - 20, y1 - 20, x2 + 20, y2 + 20], dtype=np.int32)
                else:
                    """
                    重なりチップ
                    """
                    list_rectangle = np.vstack((list_rectangle, base))
                    if around_chips(list_coord, base, tmp_w, tmp_h):
                        continue
                    rotate_angle = np.random.uniform(-180, 180)
                    bg = _rotate_chip(tmp, rotate_angle)
                    offset_x = np.random.randint(-(xmax - xmin) / 2, (xmax - xmin) / 2 + 1)
                    offset_y = np.random.randint(-(ymax - ymin) / 2, (ymax - ymin) / 2 + 1)
                    loc = [xmin - offset_x, ymin - offset_y, xmax - offset_x, ymax - offset_y]
                    vertices[:, 0] = vertices[:, 0] - offset_x
                    vertices[:, 1] = vertices[:, 1] - offset_y
                    rot_vertices = rotate_vertices(vertices, (center_abs[0] - offset_x, center_abs[1] - offset_y), rotate_angle - angle)
                    box = np.array([np.min(rot_vertices[:, 0]) - 20, np.min(rot_vertices[:, 1]) - 20, np.max(rot_vertices[:, 0]) + 20, np.max(rot_vertices[:, 1]) + 20], dtype=np.int32)
                img, list_rectangle = image_composition(img, bg, loc, region, list_rectangle, box, base)
        image_path = f'{save_dir}/{os.path.splitext(os.path.basename(image))[0]}.jpg'
        cv2.imwrite(image_path, img)
        list_rectangle = np.unique(list_rectangle, axis=0)
        create_annotaion_xml.create_xml(image_path, list_rectangle, save_dir, w, h)


def complement(img):
    indecies1, indecies2, indecies3 = np.where(img == 0)
    map_list = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    h, w, _ = img.shape
    for x, y, z in zip(indecies1, indecies2, indecies3):
        pix = 0
        num = 0
        for i, j in map_list:
            px = img[min(max(x + i, 0), h - 1), min(max(y + j, 0), w - 1), z]
            if px > 0:
                num += 1
                pix += px
        img[x, y, z] = pix / num if num > 0 else np.random.randint(48, 52)

    return img


def read_text(path):
    with open(path, 'r', encoding='shift_jis') as f:
        line = f.readlines()
    dir_path = line[0]
    model_number = line[1]
    dir_name = dir_path.split('\\')[-2]
    image_path = "/mnt/hqnas-ai-img2-data/B238_NgImage/202101_Mapping/画像取得(技術サンプル)/暫定処置_量産品/{:}/0001_0_0".format(
        dir_name)
    model_number = model_number.split('.')[0]
    return image_path, model_number
