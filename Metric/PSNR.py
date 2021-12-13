import torch
import numpy as np

# Reference: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python


class PSNR:
    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean(np.power(img1 - img2, 2))
        return 10 * torch.log10(1. / mse).item()
