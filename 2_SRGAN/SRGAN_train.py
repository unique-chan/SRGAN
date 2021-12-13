import sys

sys.path.append('../')

import datetime

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset.dataset import LR_HR_PairDataset
from Model.Generator import Generator
from Model.Discriminator import Discriminator
from SRGAN_parser import Parser
from SRGAN_utils import *

if __name__ == '__main__':
    # Parser
    my_parser = Parser(mode='train')
    my_args = my_parser.parser.parse_args()

    # Tag
    cur_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    tag_name = f'{my_args.tag}-{cur_time}' if my_args.tag else f'{cur_time}'
    print('[SRGAN]')
    print(f'{tag_name} experiment has been started.')

    # Parameter
    train_dir = my_args.train_dir
    valid_dir = my_args.valid_dir
    device = f'cuda:{my_args.gpu_index}' if my_args.gpu_index >= 0 else 'cpu'

    # Loader (Train / Valid)
    train_dataset = LR_HR_PairDataset(train_dir, mode='train', crop_size=128)
    valid_dataset = LR_HR_PairDataset(valid_dir, mode='eval', crop_size=256)
    train_loader = DataLoader(train_dataset, my_args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, my_args.batch_size, shuffle=False, pin_memory=True)

    discriminator = Discriminator(in_channels=3)
    discriminator_best_psnr_model_state = discriminator.state_dict()
    discriminator_best_ssim_model_state = discriminator.state_dict()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=my_args.d_lr, betas=(0.9, 0.999))

    generator = Generator(in_channels=3, scaling_factor=4, num_residual_blocks=16)
    generator.load_state_dict(torch.load(my_args.generator_pt_path))
    generator_best_psnr_model_state = generator.state_dict()
    generator_best_ssim_model_state = generator.state_dict()
    g_optimizer = optim.Adam(generator.parameters(), lr=my_args.g_lr, betas=(0.9, 0.999))

    db_valid_PSNR_SSIM = {'psnr': [], 'ssim': []}
    tensorboard_writer = SummaryWriter(f'./log_dir/{tag_name}')

    for epoch in range(my_args.epochs):
        valid_psnr, valid_ssim = \
            train_and_validate_SRGAN(generator, g_optimizer, discriminator, d_optimizer, epoch, device,
                                     train_loader, valid_loader, my_args.content_loss_function, tensorboard_writer)
        if db_valid_PSNR_SSIM['psnr'] and \
                valid_ssim > max(db_valid_PSNR_SSIM['psnr']):  # store the best model so far.
            generator_best_psnr_model_state = generator.state_dict()
            discriminator_best_psnr_model_state = discriminator.state_dict()
        if db_valid_PSNR_SSIM['ssim'] and \
                valid_ssim > max(db_valid_PSNR_SSIM['ssim']):  # store the best model so far.
            generator_best_ssim_model_state = generator.state_dict()
            discriminator_best_ssim_model_state = discriminator.state_dict()

        db_valid_PSNR_SSIM['psnr'].append(valid_psnr)
        db_valid_PSNR_SSIM['ssim'].append(valid_ssim)

    generator_last_model_state = generator.state_dict()
    discriminator_last_model_state = discriminator.state_dict()

    torch.save(generator_best_psnr_model_state, f'SRGAN-generator-best-psnr-{tag_name}.pt')
    torch.save(generator_best_ssim_model_state, f'SRGAN-generator-best-ssim-{tag_name}.pt')
    torch.save(generator_last_model_state, f'SRGAN-generator-last-{tag_name}.pt')
    torch.save(discriminator_best_psnr_model_state, f'SRGAN-discriminator-best-psnr-{tag_name}.pt')
    torch.save(discriminator_best_ssim_model_state, f'SRGAN-discriminator-best-ssim-{tag_name}.pt')
    torch.save(discriminator_last_model_state, f'SRGAN-discriminator-last-{tag_name}.pt')
    print(f'SRGAN-generator-best-psnr-{tag_name}.pt is stored.')
    print(f'SRGAN-generator-best-ssim-{tag_name}.pt is stored.')
    print(f'SRGAN-generator-last-psnr-{tag_name}.pt is stored.')
    print(f'SRGAN-generator-last-ssim-{tag_name}.pt is stored.')
    print(f'SRGAN-discriminator-best-psnr-{tag_name}.pt is stored.')
    print(f'SRGAN-discriminator-best-ssim-{tag_name}.pt is stored.')
    print(f'SRGAN-discriminator-last-psnr-{tag_name}.pt is stored.')
    print(f'SRGAN-discriminator-last-ssim-{tag_name}.pt is stored.')
    write_log_for_SRGAN(db_valid_PSNR_SSIM, f'SRGAN-{tag_name}-log.txt')
    print(f'SRGAN-{tag_name}-log.txt is stored.')
    plot_log_for_SRGAN(db_valid_PSNR_SSIM, f'SRGAN-{tag_name}-log.png')
    print(f'SRGAN-{tag_name}-log.png is stored.')
