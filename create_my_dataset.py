import glob
import random
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="hyper-parameters for create my PATH for diffusion SR", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--PATH", default="/home/yuan/Downloads/PA_dataset/MA", type=str)
parser.add_argument("--TargetPATH_HR", default="/media/yuan/软件盘/YUAN_NET/DiSR/dataset/", type=str)
parser.add_argument("--TargetPATH_LR", default="/media/yuan/软件盘/YUAN_NET/DiSR/dataset/", type=str)
parser.add_argument("--TargetPATH_SR", default="/media/yuan/软件盘/YUAN_NET/DiSR/dataset/", type=str)
parser.add_argument("--sub_train", default="RealTrain", type=str)
parser.add_argument("--sub_val", default="RealValid", type=str)
parser.add_argument("--sub_test", default="test", type=str)
parser.add_argument("--target_size", default=256, type=int)

parser.add_argument("--mode", default='for_test', choices=["for_train", "for_val", "fro_test"], type=str)

args = parser.parse_args()
# if args.mode == "for_train":
#     img_paths = glob.glob(os.path.join(args.PATH, args.sub_train, "*/*.*"))
#     target_path_hr = os.path.join(args.TargetPATH_HR, "train_PA/hr_128")
#     target_path_lr = os.path.join(args.TargetPATH_HR, "train_PA/lr_16")
#     target_path_sr = os.path.join(args.TargetPATH_HR, "train_PA/sr_16_128")
# elif args.mode == "for_val":
#     img_paths = glob.glob(os.path.join(args.PATH, args.sub_val, "*/*.*"))
#     target_path_hr = os.path.join(args.TargetPATH_HR, "val_PA/hr_128")
#     target_path_lr = os.path.join(args.TargetPATH_HR, "val_PA/lr_16")
#     target_path_sr = os.path.join(args.TargetPATH_HR, "val_PA/sr_16_128")
# elif args.mode == "for_test":
#     img_paths = glob.glob(os.path.join(args.PATH, args.sub_test, "*/*.*"))
#     target_path_hr = os.path.join(args.TargetPATH_HR, "test_PA/hr_128")
#     target_path_lr = os.path.join(args.TargetPATH_HR, "test_PA/lr_16")
#     target_path_sr = os.path.join(args.TargetPATH_HR, "test_PA/sr_16_128")
#
if args.mode == "for_train":
    img_paths = glob.glob(os.path.join(args.PATH, args.sub_train, "*/*.*"))
    target_path_hr = os.path.join(args.TargetPATH_HR, "train_PA/hr_256")
    target_path_lr = os.path.join(args.TargetPATH_HR, "train_PA/lr_256")
    target_path_sr = os.path.join(args.TargetPATH_HR, "train_PA/sr_256_256")
elif args.mode == "for_val":
    img_paths = glob.glob(os.path.join(args.PATH, args.sub_val, "*/*.*"))
    target_path_hr = os.path.join(args.TargetPATH_HR, "val_PA/hr_256")
    target_path_lr = os.path.join(args.TargetPATH_HR, "val_PA/lr_256")
    target_path_sr = os.path.join(args.TargetPATH_HR, "val_PA/sr_256_256")
elif args.mode == "for_test":
    img_paths = glob.glob(os.path.join(args.PATH, args.sub_test, "*/*.*"))
    target_path_hr = os.path.join(args.TargetPATH_HR, "test_PA/hr_256")
    target_path_lr = os.path.join(args.TargetPATH_HR, "test_PA/lr_256")
    target_path_sr = os.path.join(args.TargetPATH_HR, "test_PA/sr_256_256")
# random crop to 128 and PyrDown to 16
N = 32
if not os.path.exists(target_path_sr):
    os.makedirs(target_path_sr)
if not os.path.exists(target_path_lr):
    os.makedirs(target_path_lr)
if not os.path.exists(target_path_hr):
    os.makedirs(target_path_hr)
for idx in range(N):
    random_num = random.randint(0, len(img_paths)-1)
    img_path = img_paths[random_num]
    raw_img = cv2.imread(img_path)
    h, w, c = raw_img.shape
    # random choose upper-left cornet
    if h >= args.target_size and w>=args.target_size:
        upper_left_y = random.randint(0, h - args.target_size)
        upper_left_x = random.randint(0, w - args.target_size)
    cropped_img = raw_img[upper_left_y:upper_left_y+args.target_size, upper_left_x:upper_left_x+args.target_size, :]
    cv2.imwrite(os.path.join(target_path_hr, str(idx)+".jpg"), cropped_img)
    # # downsampling
    # cropped_img_lr = cv2.resize(cropped_img, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(target_path_lr, str(idx) + ".jpg"), cropped_img_lr)
    # # downsampling
    # cropped_img_lr_interp = cv2.resize(cropped_img_lr, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(target_path_sr, str(idx) + ".jpg"), cropped_img_lr_interp)
    # downsampling
    cropped_img_lr = cv2.resize(cropped_img, (0, 0), fx=1/8, fy=1/8, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(target_path_lr, str(idx) + ".jpg"), cropped_img_lr)
    # downsampling
    cropped_img_lr_interp = cv2.resize(cropped_img_lr, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(target_path_sr, str(idx) + ".jpg"), cropped_img_lr_interp)
    cv2.imwrite(os.path.join(target_path_lr, str(idx) + ".jpg"), cropped_img_lr_interp)

