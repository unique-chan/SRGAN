import argparse
import os
from pathlib import Path
import PIL
from PIL import Image

import glob
import tqdm

parser = argparse.ArgumentParser(description='high-resolution to low-resolution')
parser.add_argument('--HR_dir', type=str)
args = parser.parse_args()

HR_dir = Path(f'{args.HR_dir}')
HR_train_dir = Path(f'{HR_dir}/train')
HR_valid_dir = Path(f'{HR_dir}/valid')

print(f'[Opened]\tHR_dir: {HR_dir}\n\t\tHR_train_dir: {HR_train_dir}\n\t\tHR_valid_dir: {HR_valid_dir}')

LR_dir = Path(f'{HR_dir.parent}/LR')
LR_train_dir = Path(f'{LR_dir}/train')
LR_valid_dir = Path(f'{LR_dir}/valid')

rescale_ratio = 4

os.makedirs(LR_dir, exist_ok=True)
os.makedirs(LR_train_dir, exist_ok=True)
os.makedirs(LR_valid_dir, exist_ok=True)

print(f'[Created]\tLR_dir: {LR_dir}\n\t\tLR_train_dir: {LR_train_dir}\n\t\tLR_valid_dir: {LR_valid_dir}')

LR_dirs = [LR_train_dir, LR_valid_dir]

for i, cur_dir in enumerate([HR_train_dir, HR_valid_dir]):
    HR_p_dir = cur_dir
    HR_files_full_path = sorted(glob.glob(f'{cur_dir}/*.*'))
    HR_files = sorted(os.listdir(f'{cur_dir}/'))

    LR_p_dir = LR_dirs[i]

    tqdm_loader = tqdm.tqdm(HR_files_full_path, mininterval=0.1)
    for j, HR_file_full_path in enumerate(tqdm_loader):
        hr_img = Image.open(HR_file_full_path)

        hr_w, hr_h = hr_img.size
        lr_w, lr_h = hr_w // rescale_ratio, hr_h // rescale_ratio

        lr_img = hr_img.resize((lr_w, lr_h), PIL.Image.ANTIALIAS)
        lr_img.save(f'{LR_p_dir}/{HR_files[j]}')