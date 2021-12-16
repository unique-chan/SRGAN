import time

import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt

from ContentLoss.ResNetLoss import ResNetLoss
from ContentLoss.VGGLoss import VGGLoss


def write_log_for_SRResNet(db_losses, file_name):
    f = open(file_name, 'w')
    f.write('write_log_for_SRResNet() \n\n')
    f.write(f"(1) training loss per each epoch ->\n{str(db_losses['train'])} \n\n")
    f.write(f"(2) validation loss per each epoch ->\n{str(db_losses['valid'])} \n")
    f.close()


def plot_log_for_SRResNet(db_losses, file_name):
    plt.clf()
    plt.title(file_name)
    plt.plot(range(1, len(db_losses['train']) + 1), np.array(db_losses['train']), label='train_mse')
    plt.plot(range(1, len(db_losses['valid']) + 1), np.array(db_losses['valid']), label='valid_mse')
    plt.legend()
    plt.savefig(file_name)


def train_and_validate_generator(generator, g_optimizer, epoch, device,
                                 train_loader, valid_loader, loss_function_name, tensorboard_writer):
    generator.to(device)
    loss_function_1 = nn.MSELoss().to(device)
    loss_function_name_1 = loss_function_name.split('+')[0]
    if loss_function_name_1 == 'vgg_loss_19_5_4':
        loss_function_1 = VGGLoss(i=5, j=4, device=device, vgg_model_name='vgg19_bn')
    elif loss_function_name_1 == 'res_loss_18_4':
        loss_function_1 = ResNetLoss(i=4, device=device, resnet_model_name='resnet18')
    elif loss_function_name_1 == 'res_loss_34_3':
        loss_function_1 = ResNetLoss(i=3, device=device, resnet_model_name='resnet34')
    elif loss_function_name_1 == 'res_loss_34_5':
        loss_function_1 = ResNetLoss(i=5, device=device, resnet_model_name='resnet34')

    loss_function_2 = None
    if len(loss_function_name.split('+')) == 2:
        loss_function_name_2 = loss_function_name.split('+')[1]
        if loss_function_name_2 == 'vgg_loss_19_5_4':
            loss_function_2 = VGGLoss(i=5, j=4, device=device, vgg_model_name='vgg19_bn')
        elif loss_function_name_2 == 'res_loss_18_4':
            loss_function_2 = ResNetLoss(i=4, device=device, resnet_model_name='resnet18')
        elif loss_function_name_2 == 'res_loss_34_3':
            loss_function_2 = ResNetLoss(i=3, device=device, resnet_model_name='resnet34')
        elif loss_function_name_2 == 'res_loss_34_5':
            loss_function_2 = ResNetLoss(i=5, device=device, resnet_model_name='resnet34')

    # (1) train phase
    train_mse = 0
    generator.train()
    train_tqdm_loader = tqdm.tqdm(train_loader)
    for i, (lr_img, hr_img) in enumerate(train_tqdm_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        sr_img = generator(lr_img)
        train_loss = loss_function_1(hr_img, sr_img)
        if loss_function_2:
            train_loss = train_loss * 0.7
            train_loss += loss_function_2(hr_img, sr_img) * 0.3
        train_mse += train_loss.item()

        # optimization
        g_optimizer.zero_grad()
        train_loss.backward()
        g_optimizer.step()

        train_tqdm_loader.set_description(f'Train-SRResNet | Epoch: {epoch + 1} | Loss: {train_mse / (i + 1): .4f}')

        # tensorboard log
    train_mse /= len(train_tqdm_loader)
    tensorboard_writer.add_scalar('Loss/train', train_mse, epoch + 1)
    time.sleep(0.01)

    # (2) validation phase
    valid_mse = 0
    generator.eval()
    valid_tqdm_loader = tqdm.tqdm(valid_loader)
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(valid_tqdm_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr_img = generator(lr_img)
            valid_loss = loss_function_1(hr_img, sr_img)
            valid_mse += valid_loss.item()

            valid_tqdm_loader.set_description(f'Valid-SRResNet | Epoch: {epoch + 1} | '
                                              f'Loss: {valid_mse / (i + 1): .4f}')

    valid_mse /= len(valid_tqdm_loader)
    tensorboard_writer.add_scalar('Loss/valid', valid_mse, epoch + 1)

    return train_mse, valid_mse
