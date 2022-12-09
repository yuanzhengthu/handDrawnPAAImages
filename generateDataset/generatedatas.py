import os
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import cv2
import glob
import numpy as np
import argparse
from MainWidget import rain_blur, get_noise
parser = argparse.ArgumentParser()
# parser.add_argument('--dir_d1', type=str, default='/media/yuan/SOLID/20220428_hit/device01/')
parser.add_argument('--dir', type=str, default='E:\\YUANDISK_PERSONAL\\321output\\')
parser.add_argument('--crop_height', type=int, default=256)
parser.add_argument('--crop_width', type=int, default=256)
parser.add_argument('--total_num_generate', type=str, default=2)
args = parser.parse_args()
mydir = args.dir
total_num_generate = args.total_num_generate
crop_height = args.crop_height
crop_width = args.crop_width
dir_saving = os.path.join(args.dir, 'noise_imgs5')
if not os.path.exists(dir_saving):
    os.makedirs(dir_saving)

# for idx in ['train', 'valid', 'test']:
#     if not os.path.exists(os.path.join(dir_saving, idx)):
#         os.makedirs(os.path.join(dir_saving, idx))

img_paths = glob.glob(os.path.join(mydir, "*.png"))
img_paths.sort(key=lambda x: int(x.split('\\')[-1].split(".")[0]))
total_imgs = len(img_paths)
# for sel in np.arange(0, total_num_generate):
#     for idx in np.arange(0, total_imgs):
#         # for sp_point_x in np.arange(0, 256):
#         #     for sp_point_y in np.arange(0, 256):
#         sp_point_x = np.random.randint(0, 144)
#         sp_point_y = np.random.randint(0, 144)
#         img = cv2.imread(img_paths[idx])
#         crop_img = img[sp_point_x:sp_point_x+crop_height, sp_point_y:sp_point_y+crop_width, :]
#         crop_img = cv2.resize(crop_img, (32, 32), interpolation=cv2.INTER_CUBIC)
#         crop_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite(
#             os.path.join(dir_saving, 'test_raw_' + str(idx) + '_' + str(sp_point_x) + '_' + str(sp_point_y) + '.jpg'),
#             crop_img)
#         for idy in range(2):
#             noise_value = np.random.randint(20, 80)
#             rain_length = np.random.randint(10, 50)
#             rain_angular = np.random.randint(-60, 60)
#             rain_width = np.random.randint(2, 5) * 2 + 1
#             noise = get_noise(crop_img, value=noise_value)
#             rain = rain_blur(noise, length=rain_length, angle=-rain_angular, w=rain_width)
#             crop_img = cv2.blur(crop_img + cv2.cvtColor(rain, cv2.COLOR_GRAY2RGB), (3, 3))
#             cv2.imwrite(
#                 os.path.join(dir_saving,
#                              'test_rain_' + str(idy) + str(idx) + '_' + str(sp_point_x) + '_' + str(sp_point_y) + '.jpg'),
#                 crop_img)
#             #crop_img = cv2.blur(crop_img + cv2.cvtColor(rain, cv2.COLOR_GRAY2RGB), (10, 10))
#             #crop_img = cv2.GaussianBlur(crop_img, (7, 7), 10)
#
#         cv2.imwrite(os.path.join(dir_saving, 'test_'+str(idx)+'_'+str(sp_point_x)+'_'+str(sp_point_y)+'.jpg'), crop_img)

for sel in np.arange(0, total_num_generate):
    for idx in np.arange(0, total_imgs):
        # for sp_point_x in np.arange(0, 256):
        #     for sp_point_y in np.arange(0, 256):
        sp_point_x = np.random.randint(0, 144)
        sp_point_y = np.random.randint(0, 144)
        img = cv2.imread(img_paths[idx])
        crop_img = img[sp_point_x:sp_point_x+crop_height, sp_point_y:sp_point_y+crop_width, :]
        crop_img = cv2.resize(crop_img, (32, 32), interpolation=cv2.INTER_CUBIC)
        crop_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(
            os.path.join(dir_saving, 'test_raw_' + str(idx) + '_' + str(sp_point_x) + '_' + str(sp_point_y) + '.jpg'),
            crop_img)
        for idy in range(2):
            noise_value = np.random.randint(0, 50)
            rain_length = np.random.randint(10, 80)
            rain_angular = np.random.randint(-60, 60)
            rain_width = np.random.randint(2, 5) * 2 + 1
            noise = get_noise(crop_img, value=noise_value)
            rain = rain_blur(noise, length=rain_length, angle=-rain_angular, w=rain_width)
            crop_img = cv2.blur(crop_img + cv2.cvtColor(rain, cv2.COLOR_GRAY2RGB), (3, 3))
            cv2.imwrite(
                os.path.join(dir_saving,
                             'test_rain_' + str(idy) + str(idx) + '_' + str(sp_point_x) + '_' + str(sp_point_y) + '.jpg'),
                crop_img)
            #crop_img = cv2.blur(crop_img + cv2.cvtColor(rain, cv2.COLOR_GRAY2RGB), (10, 10))
            #crop_img = cv2.GaussianBlur(crop_img, (7, 7), 10)

        cv2.imwrite(os.path.join(dir_saving, 'test_'+str(idx)+'_'+str(sp_point_x)+'_'+str(sp_point_y)+'.jpg'), crop_img)
