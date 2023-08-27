import torch
from kornia.geometry import transform as K
import cv2
import numpy as np
from typing import Dict
import glob
import tqdm
import os
import jpeg4py


def rotate_affine(imgs, angles, length, device):
    b, c, h, w = imgs.shape
    dsts = torch.zeros((b, c, length, length), device=device)
    dsts[:, :, (length - h) // 2:(length - h) // 2 + h, (length - w) // 2:(length - w) // 2 + w] = imgs
    angles = angles.to(device=device)
    M = K.get_rotation_matrix2d(
        torch.tensor([length / 2, length / 2], device=device).repeat(b, 1),
        angles,
        torch.tensor([1.0, 1.0], device=device).repeat(b, 1))
    return K.warp_affine(dsts, M, dsize=(length, length))


def rotate_affine_cv2(image, angle, length):
    # type: (np.ndarray, float, int) -> np.ndarray
    h, w = image.shape[:2]
    dst = np.zeros((length, length, 3), dtype=np.uint8)
    dst[(length - h) // 2: (length - h) // 2 + h, (length - w) // 2:(length - w) // 2 + w, :] = image
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    rot_img = cv2.warpAffine(dst, trans, (length, length))
    return rot_img


def get_rotated_img_dict(basepath, ext, save_dir):
    # type: (str, str, str) -> Dict[str, np.ndarray]
    list_image = glob.glob(f'{basepath}/*{ext}')
    pbar = tqdm.tqdm(list_image)
    pbar.set_description('rotate images')
    dict_image_array = {}
    for image in pbar:
        basename = os.path.basename(image)
        items = os.path.splitext(basename)[0].split('_')
        angle = float(items[3]) % 360 * (-1)
        img = jpeg4py.JPEG(image).decode()
        if np.max(img) < 100:
            continue
        h, w = img.shape[:2]
        length = int((h ** 2 + w ** 2) ** 0.5)
        rotated_image = rotate_affine_cv2(img, angle, length)
        path = os.path.join(save_dir, basename)
        dict_image_array[path] = rotated_image
    return dict_image_array
