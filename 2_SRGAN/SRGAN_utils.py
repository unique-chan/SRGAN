import time

import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt

from ContentLoss.ResNetLoss import ResNetLoss
from ContentLoss.VGGLoss import VGGLoss
from Metric.PSNR import PSNR
import Metric.pytorch_msssim as torch_SSIM


def write_log_for_SRGAN(db_valid_PSNR_SSIM, file_name):
    f = open(file_name, 'w')
    f.write('write_log_for_SRGAN() \n\n')
    f.write(f"(1) valid PSNR per each epoch ->\n{str(db_valid_PSNR_SSIM['psnr'])} \n\n")
    f.write(f"(2) valid SSIM per each epoch ->\n{str(db_valid_PSNR_SSIM['ssim'])} \n")
    f.close()


def plot_log_for_SRGAN(db_valid_PSNR_SSIM, file_name):
    plt.clf()
    plt.title(file_name)
    plt.plot(range(1, len(db_valid_PSNR_SSIM['psnr']) + 1), np.array(db_valid_PSNR_SSIM['psnr']), label='valid_PSNR')
    plt.plot(range(1, len(db_valid_PSNR_SSIM['ssim']) + 1), np.array(db_valid_PSNR_SSIM['ssim']), label='valid_SSIM')
    plt.legend()
    plt.savefig(file_name)


def train_and_validate_SRGAN(generator, g_optimizer, discriminator, d_optimizer, epoch, device,
                                 train_loader, valid_loader, content_loss_function_name, tensorboard_writer):
    generator.to(device)
    discriminator.to(device)

    bce_loss_function = nn.BCELoss().to(device)

    content_loss_function = nn.MSELoss().to(device)
    if content_loss_function_name == 'vgg_loss_19_5_4':
        content_loss_function = VGGLoss(i=5, j=4, device=device, vgg_model_name='vgg19_bn')
    elif content_loss_function_name == 'res_loss_18_4':
        content_loss_function = ResNetLoss(i=4, device=device, resnet_model_name='resnet18')
    elif content_loss_function_name == 'res_loss_34_3':
        content_loss_function = ResNetLoss(i=3, device=device, resnet_model_name='resnet34')
    elif content_loss_function_name == 'res_loss_34_5':
        content_loss_function = ResNetLoss(i=5, device=device, resnet_model_name='resnet34')

    psnr_metric = PSNR()
    ssim_metric = torch_SSIM.MSSSIM()

    # (1) train phase
    train_d_loss, train_g_loss = 0, 0

    discriminator.train()
    generator.train()
    train_tqdm_loader = tqdm.tqdm(train_loader)
    for i, (lr_img, hr_img) in enumerate(train_tqdm_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        # (1-1) train a discriminator model.
        d_optimizer.zero_grad()
        sr_img = generator(lr_img).detach()     # detach() <- for not back-propagation
        d_i_hr = discriminator(hr_img).flatten()
        d_i_sr = discriminator(sr_img).flatten()
        d_loss_real = bce_loss_function(d_i_hr, torch.ones_like(d_i_hr))    # Goal: D[I^{HR}]    ~= 1 (for real image)
        d_loss_fake = bce_loss_function(d_i_sr, torch.zeros_like(d_i_sr))   # Goal: D[G(I^{LR})] ~= 0 (for fake image)
        d_loss = d_loss_real + d_loss_fake      # maximize      E[log[D[I^{HR}]]] + E[log[(1 - D[G(I^{LR})])]]
                                                # for PyTorch implementation (PyTorch only has a minimizer!!!),
                                                # <=> minimize  E[- log[D[I^{HR}]]] + E[- log[1 - D[G(I^{LR})]]]
        train_d_loss += d_loss.item()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        # (1-2) train a generator model.
        g_optimizer.zero_grad()
        hr_img = hr_img.detach()                # detach() <- for not back-propagation
        sr_img = generator(lr_img)
        d_i_sr = d_i_sr.detach()
        content_loss = content_loss_function(hr_img, sr_img)
        adversarial_loss = bce_loss_function(d_i_sr, torch.ones_like(d_i_sr))  # minimize  E[log[(1 - D[G(I^{LR})])]]
                                                                               # <=> minimize  - E[log[D[G(I^{LR})]]]
        g_loss = 1 * content_loss + 1e-3 * adversarial_loss

        train_g_loss += g_loss.item()
        g_loss.backward()
        g_optimizer.step()

        train_tqdm_loader.set_description(f'Train-SRGAN | Epoch: {epoch + 1} | '
                                          f'D_Loss: {train_d_loss / (i + 1): .4f} | '
                                          f'G_Loss: {train_g_loss / (i + 1): .4f} | ')

    time.sleep(0.01)

    # (2) validation phase
    generator.eval()
    valid_psnr = 0      # (2-1) PSNR
    valid_ssim = 0      # (2-2) SSIM
    valid_tqdm_loader = tqdm.tqdm(valid_loader)
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(valid_tqdm_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr_img = generator(lr_img)
            # (2-1) PSNR
            psnr = psnr_metric(hr_img, sr_img)
            valid_psnr += psnr.item()

            # (2-2) SSIM
            ssim = ssim_metric(hr_img, sr_img)
            valid_ssim += ssim.item()

            valid_tqdm_loader.set_description(f'Valid-SRGAN | Epoch: {epoch + 1} | '
                                              f'PSNR: {psnr / (i + 1): .4f} | '
                                              f'PSNR: {psnr / (i + 1): .4f} | ')
        valid_psnr /= len(valid_tqdm_loader)
        valid_ssim /= len(valid_tqdm_loader)
        tensorboard_writer.add_scalar('PSNR/valid', valid_psnr, epoch + 1)
        tensorboard_writer.add_scalar('SSIM/valid', valid_ssim, epoch + 1)

    return valid_psnr, valid_ssim
