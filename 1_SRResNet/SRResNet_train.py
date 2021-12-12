import sys
sys.path.append('../')

import datetime

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset.dataset import LR_HR_PairDataset
from Model.Generator import Generator
from SRResNet_parser import Parser
from SRResNet_utils import *


if __name__ == '__main__':
    # Parser
    my_parser = Parser(mode='train')
    my_args = my_parser.parser.parse_args()
    print('*', my_args)

    # Tag
    cur_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    tag_name = f'{my_args.tag}-{cur_time}' if my_args.tag else f'{cur_time}'
    print('[SRResNet]')
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

    # Train Generator
    # if my_args.train_generator:
    print('[Train a generator]')
    generator = Generator(in_channels=3, scaling_factor=4, num_residual_blocks=16)
    generator_best_model_state = generator.state_dict()
    g_optimizer = optim.Adam(generator.parameters(), lr=my_args.g_lr, betas=(0.9, 0.999))
    db_losses = {'train': [], 'valid': []}
    tensorboard_writer = SummaryWriter(f'./log_dir/{tag_name}')
    for epoch in range(my_args.epochs):
        train_mse, valid_mse = \
            train_and_validate_generator(generator, g_optimizer, epoch, device,
                                         train_loader, valid_loader, my_args.loss_function, tensorboard_writer)
        if db_losses['valid'] and \
                valid_mse < min(db_losses['valid']):  # store the best model so far.
            generator_best_model_state = generator.state_dict()
        db_losses['train'].append(train_mse)
        db_losses['valid'].append(valid_mse)
    torch.save(generator_best_model_state, f'SRResNet-generator-{tag_name}.pt')
    print(f'SRResNet-generator-{tag_name}.pt is stored.')
    write_log_for_SRResNet(db_losses, f'SRResNet-generator-{tag_name}-log.txt')
    print(f'SRResNet-generator-{tag_name}-log.txt is stored.')
    plot_log_for_SRResNet(db_losses, f'SRResNet-generator-{tag_name}-log.png')
    print(f'SRResNet-generator-{tag_name}-log.png is stored.')