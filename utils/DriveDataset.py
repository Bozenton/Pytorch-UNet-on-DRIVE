import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str,
                 scale: float = 1.0,
                 mask_suffix: str = '_mask'):
        assert os.path.exists(images_dir), "{} path does not exist".format(images_dir)
        assert os.path.exists(masks_dir), "{} path does not exist".format(masks_dir)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # get the images name without the kind
        self.images_name = [os.path.splitext(file)[0] for file
                            in os.listdir(images_dir) if not file.startswith('.')]
        if not self.images_name:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.images_name)} images')

    def __len__(self):
        return len(self.images_name)

    @staticmethod
    def load(filename):
        ext = os.path.splitext(filename)[1]
        if ext in ['.tif', '.gif']:
            return Image.open(filename)
        else:
            raise TypeError('The kind of image or mask file should be tif or gif')

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, f'Scale is too small, resize images would have no pixel'
        pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))  # in pytorch: [c, w, h]
        img_ndarray = img_ndarray / 255
        return img_ndarray

    def __getitem__(self, idx):
        img_name = self.images_name[idx]
        img_id = img_name.split('_')[0]

        # Iterate over the folder and yield all existing files (of any kind, including directories)
        # matching the given relative pattern
        img_file = list(self.images_dir.glob(img_name + '.*'))
        mask_file = list(self.masks_dir.glob(str(img_id) + self.mask_suffix + '.*'))
        # example: [WindowsPath('datasets/training/images/21_training.tif')]

        assert len(img_file) == 1, f'Either no image or multiple images found for {img_name}'
        assert len(mask_file) == 1, f'Either no mask or multiple mask found for {img_name}'
        img = self.load(img_file[0])
        mask = self.load(mask_file[0])
        assert img.size == mask.size, \
            f'Image and mask of {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
