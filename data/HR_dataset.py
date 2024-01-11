# -*- coding: utf-8 -*-

from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import os
def random_crop(image, crop_shape):
    ###image 是pil读取的，crop_shape是裁剪的大小

    nw = random.randint(0, image.size[0] - crop_shape[0])  ##裁剪图像在原图像中的坐标
    nh = random.randint(0, image.size[1] - crop_shape[1])
    image_crop = image.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))

    return image_crop

def fill_image(image):
    width, height = image.size
    new_width = max(width, 256)
    new_height = max(height, 256)
    padded_image = Image.new("L", (new_width, new_height), 0)
    padded_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    return padded_image

class HRDataset(Dataset):
    def __init__(self, dataset_opt, phase):
        self.l_res = dataset_opt['l_resolution']
        self.r_res = dataset_opt['r_resolution']
        self.phase = phase
        self.hr_dir = dataset_opt['dataroot']
        self.hr_image_paths = os.listdir(self.hr_dir)

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dir, self.hr_image_paths[index])
        img_HR = Image.open(hr_path).convert("L")
        img_HR = fill_image(img_HR)


        if img_HR.size != (self.r_res, self.r_res):
            img_HR = random_crop(img_HR, [self.r_res, self.r_res])

        img_LR = img_HR.resize((self.l_res, self.l_res), Image.NEAREST)
        img_SR = img_LR.resize((self.r_res, self.r_res), Image.BICUBIC)
        img_LR = img_LR.resize((self.r_res, self.r_res), Image.NEAREST)
        [img_LR, img_SR, img_HR] = Util.transform_augment(
            [img_LR, img_SR, img_HR], split=self.phase, min_max=(-1, 1))
        return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
