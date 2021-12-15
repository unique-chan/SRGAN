from PIL import Image

import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode


class HR_to_HR_LR_PairDataset(Dataset):
    def __init__(self, data_path, mode, crop_size=128, rescale_ratio=4):
        self.img_files = sorted(glob.glob(f'{data_path}/*.*'))
        assert len(self.img_files) > 0, "No images exist in the given dataset path."
        self.LR_transforms = self.get_LR_transforms(crop_size, rescale_ratio)
        self.HR_transforms = self.get_HR_transforms(mode, crop_size)

    def get_LR_transforms(self, crop_size, rescale_ratio):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size // rescale_ratio, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def get_HR_transforms(self, mode, crop_size):
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        hr_img = self.HR_transforms(Image.open(self.img_files[idx]))
        lr_img = self.LR_transforms(hr_img)
        return lr_img, hr_img


# [OLD VERSION]
# class LR_HR_PairDataset(Dataset):
#     def __init__(self, LR_path, HR_path, mode, crop_size=96):
#         self.LR_files = sorted(glob.glob(f'{LR_path}/*'))
#         self.HR_files = sorted(glob.glob(f'{HR_path}/*'))
#         self.mode = mode    # cases: ['train' | 'eval']
#         self.crop_size = crop_size
#         assert len(self.LR_files) == len(self.HR_files), "LR and HR should have same number of images."
#         assert len(self.LR_files) > 0, "No images exist in the given dataset path."
#
#     def __getitem__(self, idx):
#         if self.mode == 'train':
#             LR_transform_list = [
#                 transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1), ratio=(0.5, 2)),
#                 transforms.RandomRotation(10),
#                 transforms.RandomHorizontalFlip(0.5),
#             ]
#             HR_transform_list = [
#                 transforms.RandomResizedCrop(self.crop_size),
#                 transforms.RandomRotation(10),
#                 transforms.RandomHorizontalFlip(0.5),
#             ]
#         else:   # self.mode == 'eval'
#             LR_transform_list = [
#                 transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1), ratio=(0.5, 2)),
#             ]
#             HR_transform_list = [
#                 transforms.RandomResizedCrop(self.crop_size),
#             ]
#
#         LR_transform_func = transforms.Compose(LR_transform_list + [transforms.ToTensor()])
#         HR_transform_func = transforms.Compose(HR_transform_list + [transforms.ToTensor()])
#
#         LR_img = LR_transform_func(Image.open(self.LR_files[idx]))
#         print(LR_img.shape)
#         HR_img = HR_transform_func(Image.open(self.HR_files[idx]))
#         print(HR_img.shape)
#
#         # LR_img = pil_to_tensor(Image.open(self.LR_files[idx]))
#         # HR_img = pil_to_tensor(Image.open(self.HR_files[idx]))
#         return LR_img, HR_img
#
#     def __len__(self):
#         return len(self.LR_files)


# class SingleDataset(Dataset):
#     def __init__(self, path):
#         self.files = glob.glob(f'{path}/*')
#
#     def __getitem__(self, idx):
#         return pil_to_tensor(Image.open(self.files[idx]))
#
#     def __len__(self):
#         return len(self.files)
