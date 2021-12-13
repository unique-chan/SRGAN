import argparse


class Parser:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='PyTorch - SRGAN '
                                                          '(Yechan Kim / Unofficial Implementation)')
        self.add_default_arguments()
        if mode == 'train':
            self.add_arguments_for_train()
        elif mode == 'test':
            self.add_arguments_for_test()
        else:  # mode == 'demo'
            self.add_arguments_for_demo()

    def add_default_arguments(self):
        self.parser.add_argument('--gpu_index', default=0, type=int,
                                 help="gpu index: [-1|0|1|2|...] ([-1]='cpu mode', [>=0]:'gpu slot number')")
        self.parser.add_argument('--batch_size', default=1, type=int,
                                 help='mini-batch size (default: 1)')
        self.parser.add_argument('--generator_pt_path', type=str)

    def add_arguments_for_train(self):
        self.parser.add_argument('--d_lr', default=1e-4, type=float,
                                 help='initial learning rate '
                                      'for training a discriminator (default: 0.0001)')
        self.parser.add_argument('--g_lr', default=1e-4, type=float,
                                 help='initial learning rate '
                                      'for training (fine-tuning) a generator (default: 0.0001)')
        self.parser.add_argument('--epochs', default=1, type=int,
                                 help='training epochs for generator with discriminator (default: 1)')
        self.parser.add_argument('--train_dir', type=str,
                                 help='Low-resolution dataset dir')
        self.parser.add_argument('--valid_dir', type=str,
                                 help='High-resolution dataset dir')
        self.parser.add_argument('--tag', type=str,
                                 help='tag name for current experiment')
        self.parser.add_argument('--content_loss_function', default='mse_loss', type=str,
                                 help="['mse_loss'|'vgg_loss_19_5_4'|'res_loss_18_5_4'|'res_loss_34_5_4']")

    def add_arguments_for_test(self):
        pass

    def add_arguments_for_demo(self):
        self.parser.add_argument('--input_img_path', type=str)
        self.parser.add_argument('--output_img_path', type=str)
