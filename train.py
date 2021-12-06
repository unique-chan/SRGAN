from torch import nn, optim
from Dataset.dataset import LR_HR_PairDataset
from Model.Generator import Generator
from parser import Parser
from torch.utils.data import DataLoader
import datetime
from utils_for_train import *


if __name__ == '__main__':
    # Parser
    my_parser = Parser(mode='train')
    my_args = my_parser.parser.parse_args()

    # Tag
    cur_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    tag_name = f'{my_args.tag}-{cur_time}' if my_args.tag else f'{cur_time}'
    print(f'{tag_name} experiment has been started.')

    # Parameter
    LR_train_dir_path, HR_train_dir_path = f'{my_args.LR_dir}/train', f'{my_args.HR_dir}/train'
    LR_valid_dir_path, HR_valid_dir_path = f'{my_args.LR_dir}/valid', f'{my_args.HR_dir}/valid'
    device = f'cuda:{my_args.gpu_index}' if my_args.gpu_index >= 0 else 'cpu'

    # Loader (Train / Valid)
    train_dataset = LR_HR_PairDataset(LR_train_dir_path, HR_train_dir_path)
    valid_dataset = LR_HR_PairDataset(LR_valid_dir_path, HR_valid_dir_path)
    train_loader = DataLoader(train_dataset, my_args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, my_args.batch_size, shuffle=False, pin_memory=True)

    # Train Generator
    generator = Generator(in_channels=3, scaling_factor=4, num_residual_blocks=16)
    g_optimizer = optim.Adam(generator.parameters(), lr=my_args.lr, betas=(0.9, 0.999))
    train_generator(generator, g_optimizer, my_args.g_epochs)

    # generator.to(device)
    # generator.train()
    #
    # my_args.g_epochs
    #
    # # g_tqdm_loader =
    # # for epoch in range(g_epochs):
    # #     for
    # #     psnr = train_generator(train_loader) -> optimizer
    # #
    # #     if condition:
    # #         generator_model_store
    #
    #
    # for epoch in range(d_epochs):
    #     psnr = train_discriminator(train_loader)
    #
    #     if condition:
    #         discriminator_model_store -> optimizer
    #
    # schduler_step
