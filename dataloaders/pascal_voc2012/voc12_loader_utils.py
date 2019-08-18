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
    img_size = (size_tag.find('width').text, size_tag.find('height').text, size_tag.find('depth').text)
    
    bndboxes = list()

    for obj in ann.findall('object'):
        cat = obj.find('name').text
        bndbox = obj.find('bndbox')
        x_min, y_min = bndbox.find('xmin').text, bndbox.find('ymin').text
        x_max, y_max = bndbox.find('xmax').text, bndbox.find('ymax').text
        bndboxes.append( (cat, x_min, y_min, x_max, y_max) )

    return img_size, bndboxes

def read_line(file_path, file_name, line_number):
    content = None
    
    with open('{}/{}'.format(file_path, file_name)) as fp:
        for ix,(line) in enumerate(fp):
            if ix!=line_number:
                continue
            content=line
            break
        
    return content