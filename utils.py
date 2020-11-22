import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from settings import *


def rotate_if_need(img):
    h, w, c = img.shape
    rotated = False
    if h > w*2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated = True
    return img, rotated


def resize_if_need(img, max_h, max_w):
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img


def make_img_padding_right(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = 0  # (max_w - img_w) // 2
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


def make_img_padding_around(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = (max_w - img_w) // 2
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


def write_text(file_path, text):
    with open(file_path, 'w') as file:
        file.write(text)


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, rotated = rotate_if_need(img)
    return img, rotated


def get_file_name(file_path):
    return os.path.basename(file_path).split('.')[0]


def load_images_collate_fn(batch):
    """ key-word collate_fn """
    images, indexes, rotates = [], [], []
    for sample in batch:
        img, index, rotated = sample
        images.append(img)
        indexes.append(index)
        rotates.append(rotated)
    return images, indexes, rotates 


def load_data():
    img_paths = glob(IMG_FOLDER + '/*')

    dataset = DatasetRetriever(img_paths)
    loader = DataLoader(
        dataset,
        batch_size=BS,
        num_workers=NUM_WORKERS,
        sampler=SequentialSampler(dataset),
        pin_memory=False,
        collate_fn=load_images_collate_fn,
    )

    images = []
    indexes = []
    rotates = []
    for batch in loader:
        img, index, rotated = batch
        images += img
        indexes += index
        rotates += rotated

    return images, indexes, rotates


class DatasetRetriever(Dataset):

    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img, rotated = load_image(img_path)
        img = resize_if_need(img, IMAGE_H, IMAGE_W)
        index = get_file_name(img_path)
        return img, index, rotated
