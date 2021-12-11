import time

import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt


def write_log_for_generator(db_losses, file_name):
    f = open(file_name, 'w')
    f.write('write_log_for_generator() \n\n')
    f.write(f"(1) training loss per each epoch ->\n{str(db_losses['train'])} \n\n")
    f.write(f"(2) validation loss per each epoch ->\n{str(db_losses['valid'])} \n")
    f.close()


def plot_log_for_generator(db_losses, file_name):
    plt.clf()
    plt.title(file_name)
    plt.plot(range(1, len(db_losses['train']) + 1), np.array(db_losses['train']), label='train_mse')
    plt.plot(range(1, len(db_losses['valid']) + 1), np.array(db_losses['valid']), label='valid_mse')
    plt.legend()
    plt.savefig(file_name)


def train_and_validate_generator(generator, g_optimizer, epoch, device,
                                 train_loader, valid_loader, tensorboard_writer):
    generator.to(device)
    mse_loss = nn.MSELoss().to(device)

    # (1) train phase
    train_mse = 0
    generator.train()
    train_tqdm_loader = tqdm.tqdm(train_loader)
    for i, (lr_img, hr_img) in enumerate(train_tqdm_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        sr_img = generator(lr_img)
        # print('비교', hr_img.shape, sr_img.shape)
        train_loss = mse_loss(sr_img, hr_img)
        train_mse += train_loss.item()

        # optimization
        g_optimizer.zero_grad()
        train_loss.backward()
        g_optimizer.step()

        train_tqdm_loader.set_description(f'Train-Generator | Epoch: {epoch + 1} | Loss: {train_mse / (i + 1): .4f}')

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
            valid_loss = mse_loss(sr_img, hr_img)
            valid_mse += valid_loss.item()

            valid_tqdm_loader.set_description(f'Valid-Generator | Epoch: {epoch + 1} | '
                                              f'Loss: {valid_loss / (i + 1): .4f}')

    valid_mse /= len(valid_tqdm_loader)
    tensorboard_writer.add_scalar('Loss/valid', valid_mse, epoch + 1)

    return train_mse, valid_mse


def train_discriminator():
    pass
