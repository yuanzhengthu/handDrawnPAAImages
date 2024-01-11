import json
from collections import OrderedDict
import os
from datetime import datetime
import numpy as np
from torchvision.utils import make_grid
import math

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(opt['name'], datetime.now().strftime('%y%m%d_%H%M%S')))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
        if len(gpu_list) > 1:
            opt['distributed'] = True
        else:
            opt['distributed'] = False

    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg