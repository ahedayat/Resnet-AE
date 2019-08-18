import os
import torch

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from .voc12_loader_utils import read_line


class VOC2012Loader(Dataset):
    """
        data_dir:
                |_images
                |_masks
                |_filenames.txt
        labels_address:
            Address of file that contain labels
    """

    def __init__(self, data_dir, data_transform=None, obj_area_threshold=0.):
        self.images_path = '{}/images'.format(data_dir)
        self.masks_path = '{}/masks'.format(data_dir)
        self.objs_info_path = '{}/objs_info'.format(data_dir)
        self.obj_area_threshold = obj_area_threshold

        self.file_names = [line.rstrip('\n').split()[0] for line in open(
            '{}/filenames.txt'.format(data_dir))]
        self.data_transform = data_transform
        self.labels = None

        self.preprocess()

    def preprocess(self):
        self.labels = list()
        for ix in range(len(self.file_names)):
            file_name = os.path.splitext(self.file_names[ix])[0]

            label, relative_area, _ = read_line(
                self.objs_info_path,
                '{}.txt'.format(file_name),
                0).split(' ')

            relative_area = float(relative_area)
            if relative_area < self.obj_area_threshold:
                self.file_names.remove(self.file_names[ix])
            self.file_names[ix] = (self.file_names[ix], int(label))
            if label not in self.labels:
                self.labels.append(label)

    def get_labels(self):
        return self.labels

    def __getitem__(self, ix):
        file_name, category = self.file_names[ix]
        image = Image.open('{}/{}.jpg'.format(self.images_path, file_name))
        mask = Image.open('{}/{}.png'.format(self.masks_path, file_name))
        if self.data_transform is not None:
            image = self.data_transform(image)
            mask = self.data_transform(mask)

        image = torch.tensor(np.array(image), dtype=torch.float)
        mask = torch.tensor(np.array(mask), dtype=torch.float)
        output = torch.zeros((len(self.labels)), dtype=torch.float)
        output[category] = 1.

        image = image.view(image.size()[2], image.size()[0], image.size()[1])
        # print(mask.size())
        # mask = mask.view(mask.size()[1], mask.size()[0], mask.size()[1])

        return image, mask, output

    def __len__(self):
        return len(self.file_names)
