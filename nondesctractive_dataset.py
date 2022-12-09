import glob
import random
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="hyper-parameters for create my PATH for diffusion SR", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--PATHinput", default="D:\\YUANDISK_PERSONAL\\UncompletedProjs20220625\\DiSR_wujie\\dataset\\val_nondestractiveDet\\inputs", type=str)
parser.add_argument("--PATHoutput", default="D:\\YUANDISK_PERSONAL\\UncompletedProjs20220625\\DiSR_wujie\\dataset\\val_nondestractiveDet\\outputs", type=str)
parser.add_argument("--TargetPATH", default="D:\\YUANDISK_PERSONAL\\UncompletedProjs20220625\\DiSR_wujie\\dataset", type=str)
parser.add_argument("--target_size", default=128, type=int)

parser.add_argument("--mode", default='for_val', choices=["for_val", "for_val", "fro_test"], type=str)

args = parser.parse_args()

if args.mode == "for_val":
    img_paths_input = glob.glob(os.path.join(args.PATHinput, "*.*"))
    img_paths_output = glob.glob(os.path.join(args.PATHoutput, "*.*"))
    val_input = os.path.join(args.TargetPATH, "val_NON/input_128")
    val_output = os.path.join(args.TargetPATH, "val_NON/output_128")

# random crop to 128 and PyrDown to 16
N = 59

if not os.path.exists(val_output):
    os.makedirs(val_output)
if not os.path.exists(val_input):
    os.makedirs(val_input)
for idx in range(N):
    img_path = img_paths_input[idx]
    raw_img = cv2.imread(img_path)
    img_input = cv2.resize(raw_img, (0, 0), fx=args.target_size / 150, fy=args.target_size / 150, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(val_input, str(idx) + ".jpg"), img_input)
for idx in range(N):
    img_path = img_paths_output[idx]
    raw_img = cv2.imread(img_path)
    img_output = cv2.resize(raw_img, (0, 0), fx=args.target_size / 199, fy=args.target_size / 199, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(val_output, str(idx) + ".jpg"), img_output)
