import sys

sys.path.append('../')

from torch.utils.data import DataLoader

from Dataset.dataset import LR_HR_PairDataset
from Model.Generator import Generator
from SRResNet_parser import Parser
from SRResNet_utils import *
from Metric.PSNR import PSNR
import Metric.pytorch_msssim as torch_SSIM


if __name__ == '__main__':
    # Parser
    my_parser = Parser(mode='test')
    my_args = my_parser.parser.parse_args()

    generator_pt_path = my_args.generator_pt_path
    device = f'cuda:{my_args.gpu_index}' if my_args.gpu_index >= 0 else 'cpu'
    LR_dir = my_args.LR_dir
    HR_dir = my_args.HR_dir

    print(f'generator_pt_path: {generator_pt_path}')
    print(f'LR_dir: {LR_dir}')

    # Loader (Train / Valid)
    LR_HR = LR_HR_PairDataset(LR_dir, HR_dir)
    LR_HR_loader = DataLoader(LR_HR, 1, shuffle=False, pin_memory=True)

    generator = Generator(in_channels=3, num_residual_blocks=16)
    generator.to(device)
    generator.load_state_dict(torch.load(generator_pt_path))

    psnr_metric = PSNR()
    ssim_metric = torch_SSIM.MSSSIM()

    total_psnr = 0
    total_ssim = 0
    for i, (lr_img, hr_img) in enumerate(LR_HR_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        sr_img = generator(lr_img)

        psnr = psnr_metric(hr_img, sr_img)
        ssim = ssim_metric(hr_img, sr_img)

        total_psnr += psnr
        total_ssim += ssim

        print(f'{i+1}. psnr = {psnr}, ssim = {ssim}')

    print(f'total_psnr: {total_psnr}, mean_psnr: {total_psnr / len(LR_HR_loader)}, '
          f'total_ssim: {total_ssim}, mean_ssim: {total_ssim / len(LR_HR_loader)}')
