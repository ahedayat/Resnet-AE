import xml.etree.ElementTree as ET


def read_xml_file(ann_xml_address):
    """
        img_size:
            (width, height, depth)
        objs:
            [ (cat, x_min, y_min, x_max, y_max) for all bndboxes ]
        <<return>> :
            img_size, bndboxes
    """
    ann = ET.parse(ann_xml_address).getroot()
    size_tag = ann.findall('size')[0]
    img_size = (size_tag.find('width').text, size_tag.find(
        'height').text, size_tag.find('depth').text)

    bndboxes = list()

    for obj in ann.findall('object'):
        cat = obj.find('name').text
        bndbox = obj.find('bndbox')
        x_min, y_min = bndbox.find('xmin').text, bndbox.find('ymin').text
        x_max, y_max = bndbox.find('xmax').text, bndbox.find('ymax').text
        bndboxes.append((cat, x_min, y_min, x_max, y_max))

    return img_size, bndboxes


def read_salient_obj(file_path, file_name):
    salient_obj_info = None

    with open('{}/{}'.format(file_path, file_name)) as fp:
        for line in fp:
            label, relative_area, area = line.split()
            label, relative_area, area = int(
                label), float(relative_area), int(area)
            if label == 255 or label == 0:
                continue
            if salient_obj_info is None or salient_obj_info[1] < relative_area:
                salient_obj_info = (label, relative_area, area)

    return salient_obj_info
