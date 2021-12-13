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
    def _ssim(img_x, img_y):
        c1, c2 = np.power(0.01, 2), np.power(0.03, 2)

        img_x = img_x.astype(np.float64)
        img_y = img_y.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu_x = cv2.filter2D(img_x, -1, window)[5:-5, 5:-5]
        mu_y = cv2.filter2D(img_y, -1, window)[5:-5, 5:-5]
        mu_x_sq = np.power(mu_x, 2)
        mu_y_sq = np.power(mu_y, 2)
        mu_x_mu_y = mu_x * mu_y
        sigma_x_sq = cv2.filter2D(img_x ** 2, -1, window)[5:-5, 5:-5] - mu_x_sq
        sigma_y_sq = cv2.filter2D(img_y ** 2, -1, window)[5:-5, 5:-5] - mu_y_sq
        sigma_x_y = cv2.filter2D(img_x * img_y, -1, window)[5:-5, 5:-5] - mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + c1) * (2 * sigma_x_y + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return ssim_map.mean()
