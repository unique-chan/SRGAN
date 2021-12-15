import sys

sys.path.append('../')


import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image

from Dataset.dataset import HR_to_HR_LR_PairDataset
from Model.Generator import Generator
from SRGAN_parser import Parser
from SRGAN_utils import *


if __name__ == '__main__':
    # Parser
    my_parser = Parser(mode='demo')
    my_args = my_parser.parser.parse_args()

    print('[SRGAN] - Demo')

    lr_img_path = my_args.input_img_path
    sr_img_path = my_args.output_img_path
    generator_pt_path = my_args.generator_pt_path
    device = f'cuda:{my_args.gpu_index}' if my_args.gpu_index >= 0 else 'cpu'

    lr_img = Image.open(lr_img_path)
    lr_img = transforms.ToTensor()(lr_img)
    flag = ''
    if len(lr_img.shape) == 3:      # RGB image     -> 4D Tensor
        flag = 'RGB'
        lr_img = torch.unsqueeze(lr_img, 0)
    else:                           # Gray image    -> 4D Tensor
        flag = 'Gray'
        lr_img = torch.unsqueeze(lr_img, 0)
        lr_img = torch.unsqueeze(lr_img, 0)
    lr_img = lr_img.to(device)

    generator = Generator(in_channels=3, scaling_factor=4, num_residual_blocks=16)
    generator.to(device)
    generator.load_state_dict(torch.load(generator_pt_path))

    sr_img = generator(lr_img)
    if flag == 'RGB':               # 4D Tensor     -> 3D image (RGB)
        sr_img = sr_img.squeeze()
    else:                           # 4D Tensor     -> 2D image (Gray)
        sr_img = sr_img.squeeze()
        sr_img = sr_img.squeeze()
    save_image(sr_img, sr_img_path)
