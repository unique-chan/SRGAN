from PIL import Image

import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode

pil_to_tensor = transforms.ToTensor()


class HR_to_LR_HR_PairDataset(Dataset):
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
        hr_img = self.HR_transforms(Image.open(self.img_files[idx]).convert('RGB'))
        lr_img = self.LR_transforms(hr_img)
        return lr_img, hr_img


class LR_HR_PairDataset(Dataset):
    def __init__(self, LR_path, HR_path):
        self.LR_files = sorted(glob.glob(f'{LR_path}/*'))
        self.HR_files = sorted(glob.glob(f'{HR_path}/*'))
        print('LR_files:', self.LR_files)
        assert len(self.LR_files) == len(self.HR_files), "LR and HR should have same number of images."
        assert len(self.LR_files) > 0, "No images exist in the given dataset path."

    def __len__(self):
        return len(self.LR_files)

    def __getitem__(self, idx):
        lr_img = pil_to_tensor(Image.open(self.LR_files[idx]).convert('RGB'))
        hr_img = pil_to_tensor(Image.open(self.HR_files[idx]).convert('RGB'))

        return lr_img, hr_img


# class SingleDataset(Dataset):
#     def __init__(self, path):
#         self.files = glob.glob(f'{path}/*')
#
#     def __getitem__(self, idx):
#         return pil_to_tensor(Image.open(self.files[idx]))
#
#     def __len__(self):
#         return len(self.files)
