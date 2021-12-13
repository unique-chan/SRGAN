import torch

# Reference: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python


class PSNR:
    @staticmethod
    def __call__(img_x, img_y):
        mse = torch.mean((img_x - img_y) ** 2)
        return 10 * torch.log10(1. / mse).item()
