import xml.etree.ElementTree as ET
import os


def create_xml(image_path, list_rectangle, save_dir, w, h, list_class=None):
    image_name = os.path.basename(image_path)
    folder_name = os.path.dirname(image_path)
    xml_path = os.path.join(save_dir, os.path.splitext(image_name)[0] + '.xml')
    top = ET.Element('annotation')
    folder = ET.SubElement(top, 'folder')
    folder.text = folder_name
    filename = ET.SubElement(top, 'filename')
    filename.text = image_name
    path = ET.SubElement(top, 'path')
    path.text = image_path
    source = ET.SubElement(top, 'source')
    source.text = 'UnKnown'
    size = ET.SubElement(top, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    segmented = ET.SubElement(top, 'segmented')
    segmented.text = str(0)
    for i, rect in enumerate(list_rectangle):
        obj = ET.SubElement(top, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = list_class[i] if list_class is not None else 'chip'
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(0)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(rect[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(rect[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(rect[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(rect[3])
    data = ET.ElementTree(top)
    data.write(xml_path, encoding='utf-8')


def rotated_pascal_voc(image_path, list_vertices, save_dir, w, h):
    image_name = os.path.basename(image_path)
    folder_name = os.path.dirname(image_path)
    xml_path = os.path.join(save_dir, os.path.splitext(image_name)[0] + '.xml')
    top = ET.Element('annotation')
    folder = ET.SubElement(top, 'folder')
    folder.text = folder_name
    filename = ET.SubElement(top, 'filename')
    filename.text = image_name
    path = ET.SubElement(top, 'path')
    path.text = image_path
    source = ET.SubElement(top, 'source')
    source.text = 'UnKnown'
    size = ET.SubElement(top, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    segmented = ET.SubElement(top, 'segmented')
    segmented.text = str(0)
    list_tags = ['left_top', 'right_top', 'right_bottom', 'left_bottom']
    for vertices in list_vertices:
        obj = ET.SubElement(top, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = 'chip'
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(0)
        bndbox = ET.SubElement(obj, 'bndbox')
        for i, tag in enumerate(list_tags):
            v = ET.SubElement(bndbox, tag)
            x = vertices[i][0]
            y = vertices[i][1]
            x_ = ET.SubElement(v, 'x')
            x_.text = str(x)
            y_ = ET.SubElement(v, 'y')
            y_.text = str(y)
    data = ET.ElementTree(top)
    data.write(xml_path, encoding='utf-8')
