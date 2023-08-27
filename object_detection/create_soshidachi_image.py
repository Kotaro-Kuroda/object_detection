import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import glob
import tqdm


def annotate(xml_path, sos_boxes, save_path):
    xml = ET.parse(xml_path).getroot()
    for box in sos_boxes:
        obj = ET.SubElement(xml, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = 'soshidachi'
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(0)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(box[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(box[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(box[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(box[3])
    data = ET.ElementTree(xml)
    data.write(save_path, encoding='utf-8')


def get_boxes(xml_path):
    xml = ET.parse(xml_path).getroot()
    boxes = []
    labels = []
    size = xml.find('size')
    num_objs = 0
    for obj in xml.iter('object'):
        label = obj.find('name').text.strip()
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        num_objs += 1
    return np.array(boxes), labels


def _composition(box, direction, sos, image):
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    h, w = sos.shape[:2]
    if direction == 'left':
        if w > h:
            sos = cv2.rotate(sos, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
        xmax = x1 - 1
        xmin = xmax - w
        ymin = np.random.randint(y1, y2)
        ymax = ymin + h
    elif direction == 'right':
        if w > h:
            sos = cv2.rotate(sos, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = w, h
        xmin = x2 + 1
        xmax = xmin + w
        ymin = np.random.randint(y1, y2)
        ymax = ymin + h
    elif direction == 'under':
        if h > w:
            sos = cv2.rotate(sos, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
        ymin = y2 + 1
        ymax = ymin + h
        xmin = np.random.randint(x1, x2)
        xmax = xmin + w
    else:
        if h > w:
            sos = cv2.rotate(sos, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = w, h
        ymax = y1 - 1
        ymin = ymax - h
        xmin = np.random.randint(x1, x2)
        xmax = xmin + w
    if ymin < 0 or xmin < 0 or ymax > height or xmax > width:
        return image, None
    image[ymin:ymax, xmin:xmax] = sos
    return image, [xmin, ymin, xmax, ymax]


def composition(soshidachi_images, image, xml_path, save_dir):
    boxes, labels = get_boxes(xml_path)
    img = cv2.imread(image)
    sos_boxes = []
    for i, box in enumerate(boxes):
        label = labels[i]
        if label == 'chip':
            if np.random.uniform() < 0.3:
                random_number = np.random.uniform()
                if random_number < 1 / 4:
                    direction = 'left'
                elif 1 / 4 <= random_number < 1 / 2:
                    direction = 'right'
                elif 1 / 2 <= random_number < 3 / 4:
                    direction = 'under'
                else:
                    direction = 'above'
                ind = np.random.choice(range(len(soshidachi_images)))
                img, sos_box = _composition(box, direction, soshidachi_images[ind], img)
                if sos_box is not None:
                    sos_boxes.append(sos_box)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(image)), img)
    annotate(xml_path, sos_boxes, os.path.join(save_dir, os.path.basename(xml_path)))


def main():
    soshidachi_images = glob.glob('/home/localuser/Documents/Dataset/macro/soshidachi/D0RM-AMA01-22/soshidachi/*.tif')
    soshidachi_images = [cv2.imread(image) for image in soshidachi_images]
    xml_list = glob.glob('/home/localuser/Documents/Dataset/macro/soshidachi/D0RM-AMA01-22/original/*.xml')
    save_dir = '/home/localuser/Documents/Dataset/macro/soshidachi/D0RM-AMA01-22/artificial'
    for xml_path in tqdm.tqdm(xml_list):
        image = os.path.splitext(xml_path)[0] + '.tif'
        composition(soshidachi_images, image, xml_path, save_dir)


if __name__ == "__main__":
    main()
