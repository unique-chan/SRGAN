import argparse


class Parser:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='PyTorch 2_SRGAN (Yechan Kim / Unofficial Implementation)')
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

    def add_arguments_for_train(self):
        # for training a generator!
        self.parser.add_argument('--g_lr', default=1e-4, type=float,
                                 help='initial learning rate for training a generator (default: 0.0001)')
        self.parser.add_argument('--g_epochs', default=1, type=int,
                                 help='training epochs for generator (default: 1)')
        self.parser.add_argument('--train_dir', type=str,
                                 help='Low-resolution dataset dir')
        self.parser.add_argument('--valid_dir', type=str,
                                 help='High-resolution dataset dir')
        self.parser.add_argument('--tag', type=str,
                                 help='tag name for current experiment')
        # self.parser.add_argument('--train_generator', action='store_true',
        #                          help='for training a generator!')
        # # for
        # self.parser.add_argument('--d_epochs', default=1, type=int,
        #                          help='training epochs for discriminator (default: 1)')
        #
        # # self.parser.add_argument('--store', action='store_true',
        # #                          help="store the best model during training")

    def add_arguments_for_test(self):
        pass

    def add_arguments_for_demo(self):
        self.parser.add_argument('--input_img_path', type=str)
        self.parser.add_argument('--output_img_path', type=str)
        self.parser.add_argument('--generator_pt_path', type=str)
