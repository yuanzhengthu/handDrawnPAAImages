import torch
import torchvision
import random


def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [torchvision.transforms.ToTensor()(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        if random.random() > 0.5:
            imgs = torch.flip(imgs, dims=[1])
        if random.random() > 0.5:
            imgs = torch.flip(imgs, dims=[2])
        if random.random() > 0.5:
            imgs = torch.rot90(imgs, 1, dims=[2, 3])
        if random.random() > 0.5:
            imgs = torch.rot90(imgs, 3, dims=[2, 3])
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img
