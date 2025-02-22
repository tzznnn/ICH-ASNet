from random import randint
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from torchvision.transforms import InterpolationMode



def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    indices = np.argwhere(mask != 0)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask == 0)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask != 0)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask == 0)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2]
    return pt[np.newaxis, :], [point_label]

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    shiftw = randint(-int(0.9*img_size), int(1.1*img_size))
    shifth = randint(-int(0.9*img_size), int(1.1*img_size))
    shiftx = randint(-int(0.05*img_size), int(0.05*img_size))
    shifty = randint(-int(0.05*img_size), int(0.05*img_size))

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask != 0)
    # indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

class Transform2D_BCIHM:

    def __init__(self, mode='test', img_size=256, low_img_size=256, ori_size=256):
        self.mode = mode
        self.img_size = img_size
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):

        # transforming to tensor
        image, mask = F.to_tensor(image), F.to_tensor(mask)

        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = (image - image.min()) / (image.max() - image.min())
        return image, mask, low_mask

class BCIHM(Dataset):
    def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, fold=0, img_size=256, prompt = "click", class_id=1,) -> None:
        self.fold_ = 4
        self.fold = fold
        self.dataset_path = dataset_path
        self.split = split
        id_list_file = os.path.join('./dataset/excel', 'BCIHM.csv')
        df = pd.read_csv(id_list_file, encoding='gbk')
        if self.split == 'train':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] != self.fold and df['label'][id] > 0]
            self.gt_list = [label for id, label in enumerate(df['gt']) if df['fold'][id] != self.fold and df['label'][id] > 0]
        elif self.split == 'val':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        elif self.split == 'test':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.classes = 2
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        """Get the images"""
        name = self.img_list[i]
        img_path = os.path.join(self.dataset_path, name)
        mask_name = self.gt_list[i]
        msk_path = os.path.join(self.dataset_path, mask_name)

        image = np.load(img_path)
        mask = np.load(msk_path)

        class_id = 1  # fixed since only one class of foreground
        mask[mask > 0] = 1

        image = np.clip(image, np.percentile(image, 0.05), np.percentile(image, 99.5)).astype(np.int16)

        mask = mask.astype(np.uint8)
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
            mask, low_mask = mask.squeeze(0), low_mask.squeeze(0)
         # --------- make the point prompt ----------
        if self.prompt == 'click':
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            pt = pt * self.img_size / 512
            mask[mask!=0] = 1
            mask[mask!=1] = 0
            low_mask[low_mask!=0] = 1
            low_mask[low_mask!=1] = 0
            point_labels = np.array(point_label)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        bbox = bbox * self.img_size / 512
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask': low_mask,
            'image_name': name.split('/')[-1].split('.')[0] + '.png',
            'class_id': class_id,
            }

class Transform2D_INSTANCE:

    def __init__(self, mode='test', img_size=256, low_img_size=256, ori_size=256):
        self.mode = mode
        self.img_size = img_size
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        image, mask = F.to_tensor(image), F.to_tensor(mask)
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = (image - image.min()) / (image.max() - image.min())
        return image, mask, low_mask

class INSTANCE(Dataset):
    def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, fold=0, img_size=256, prompt = "click", class_id=1,) -> None:
        self.fold = fold
        self.dataset_path = dataset_path
        self.split = split
        id_list_file = os.path.join('./dataset/excel', 'Instance.csv')
        df = pd.read_csv(id_list_file, encoding='gbk')
        if self.split == 'train':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] != self.fold and df['label'][id] > 0]
            self.gt_list = [label for id, label in enumerate(df['gt']) if df['fold'][id] != self.fold and df['label'][id] > 0]
        elif self.split == 'val':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        elif self.split == 'test':
            self.img_list = [name for id, name in enumerate(df['img']) if df['fold'][id] == self.fold]
            self.gt_list = [name for id, name in enumerate(df['gt']) if df['fold'][id] == self.fold]
        self.prompt=prompt
        self.img_size = img_size
        self.class_id = class_id
        self.classes = 2
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        """Get the images"""
        name = self.img_list[i]
        img_path = os.path.join(self.dataset_path, name)
        mask_name = self.gt_list[i]
        msk_path = os.path.join(self.dataset_path, mask_name)

        image = np.load(img_path)
        mask = np.load(msk_path)

        class_id = 1  # fixed since only one class of foreground
        mask[mask > 0] = 1

        image = np.clip(image, np.percentile(image, 0.05), np.percentile(image, 99.5)).astype(np.int16)
        mask = mask.astype(np.uint8)
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
            mask, low_mask = mask.squeeze(0), low_mask.squeeze(0)

        # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            pt = pt * self.img_size / 512
            mask[mask!=0] = 1
            mask[mask!=1] = 0
            low_mask[low_mask!=0] = 1
            low_mask[low_mask!=1] = 0
            point_labels = np.array(point_label)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        bbox = bbox * self.img_size / 512
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask':low_mask,
            'image_name': name.split('/')[-1].split('.')[0] + '.png',
            'class_id': class_id,
            }



