import numpy as np
import cv2

# Reference: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python


class SSIM:
    @staticmethod
    def __call__(img1, img2):
        assert img1.shape == img2.shape, f"The given pair of images have different dimensions, " \
                                         f"img1.shape: {img1.shape}, img2.shape: {img2.shape}"
        assert img1.ndim in [2, 3] and \
               img2.ndim in [2, 3], f"Incorrect input image dimensions (Neither Gray or RGB), " \
                                    f"img1.shape: {img1.shape}, img2.shape: {img2.shape}!"

        if img1.ndim == 2:                  # Gray (ndim == 2)
            return SSIM._ssim(img1, img2)
        else:                               # RGB  (ndim == 3)
            if img1.shape[2] == 3:
                return np.array([SSIM._ssim(img1, img2) for _ in range(3)]).mean()
            elif img1.shape[2] == 1:
                return SSIM._ssim(np.squeeze(img1), np.squeeze(img2))

    @staticmethod
    def _ssim(img1, img2):
        c1, c2 = 0.01 ** 2, 0.03 ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()
