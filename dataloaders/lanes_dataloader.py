from __future__ import print_function, division
import os,torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random,cv2


class LanesSegmentation(Dataset):
    """
    Lane segmentation dataset
    including 2 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 dataset,
                 split='train',
                 transform=None,
                 domain = None
                 ):
        """
        :param base_dir: path to lane dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.transform = transform
        self.split = split
        self.domain = domain

        self.image_pool = []
        self.label_pool = []
        self.label_pool_boundary = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split)
        with open(os.path.join(self._image_dir, split + '.txt')) as f:
            self.image_list = []
            for line in f:
                self.image_list.append(line.strip())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.domain is None:
            image_path = os.path.join(self._image_dir,'images', self.image_list[index]+'.jpg')
            image = cv2.resize(cv2.imread(image_path), (512, 256))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

            label_path = os.path.join(self._image_dir,'labels', self.image_list[index]+'.jpg')
            label = cv2.resize(cv2.imread(label_path, 0) // 255, (512, 256))
            label = torch.from_numpy(label).contiguous().long()

            # label_boundary_path = os.path.join(self._image_dir,'labels_boundary', self.image_list[index]+'.jpg')
            label_boundary = np.expand_dims(cv2.resize(cv2.imread(label_path, 0) // 255, (512, 256)),axis=0)
            label_boundary = torch.from_numpy(label_boundary).contiguous().float()

            img_name = self.image_list[index]
            anco_sample = {'image': image, 'label': label, 'boundary': label_boundary,'img_name': img_name}
        else:
            image_path = os.path.join(self._image_dir, 'images', self.image_list[index] + '.jpg')
            image = cv2.resize(cv2.imread(image_path), (512, 256))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

            img_name = self.image_list[index]
            anco_sample = {'image': image, 'img_name': img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def __str__(self):
        return 'Lanes(split=' + str(self.split) + ')'


