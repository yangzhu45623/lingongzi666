from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms as transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = transforms.ToTensor()(img_nd)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)  # mask的前景标注值都是255

        return {
            'image': img.type(torch.FloatTensor),
            'mask': mask.type(torch.FloatTensor)
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

class NewDataset(Dataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                      '.tif', '.tiff', '.webp', '.JPEG')
    def __init__(self,imgs_dir, masks_dir, scale=1, mask_suffix='') -> None:
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = self.get_ids()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        images = glob(os.path.join(self.imgs_dir,"**",'*.png'),recursive=True)
        images = list(filter(lambda x: os.path.splitext(x)[-1].lower() in self.IMG_EXTENSIONS,images))
        return images
    
    def get_mask_path(self,path):
        _dir,name = os.path.split(path)
        return os.path.splitext(name)[0]
    
    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = transforms.ToTensor()(img_nd)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = idx
        mask_file = glob(os.path.join(self.masks_dir,self.get_mask_path(img_file)+'.*'))
        assert len(mask_file) == 1, "Please check if the picture and the mask correspond!"
        mask = Image.open(mask_file[0]).transpose(Image.FLIP_TOP_BOTTOM)
        img = Image.open(img_file).convert("RGB")

        assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)  # mask的前景标注值都是255
        return {
            'image': img.type(torch.FloatTensor),
            'mask': mask.type(torch.FloatTensor)
        }