# SRGAN
## Unofficial Implementation for "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Model (C"
2021 Fall, Computer Vision (EC5303), Final Term Project
+ ID: 20212047
+ Name: Yechan Kim

### Prerequisites
- See `requirements.txt` for details.
~~~ME
torch >= 1.10.0
torchvision >= 0.11.1
glob2
numpy
matplotlib
tqdm
tensorboard
tensorboardx
~~~

### Toy-Examples
- This repository provides DIV2K (with only few examples) and Set5 for convenience. Note that given DIV2K has only few examples, thus you need to download the full DIV2K dataset for training.
- You can download DIV2K or other datasets for super-resolution task via internet.

### Novelty
- In this term project, I proposed an efficient content loss based on ResNet, not VGG-Net.
- See `Report_for_SRGAN.pdf` for details.


### SRResNet (⇔ Generator of SRGAN)
- (1) `1_SRResNet/SRResNet_train.py`
  - This source code is for training a SRResNet model.
  - How to run?
    - E.g.
      - python SRResNet_train.py --gpu_index=0 --batch_size=40 --tag='Example-SRResNet' --loss_function='MSE+res_loss_34_3' --train_dir='../Dataset/DIV2K/HR_dummy/train' --valid_dir='../Dataset/DIV2K/HR_dummy/valid' --epochs=100
    - Please see `1_SRResNet/SRResNet_parser.py` - `add_default_arguments()` & `add_arguments_for_train()` for all possible arguments to run the code.
    - Note that you only need to prepare **HR (High-Resolution) dataset**. LR (Low-Resolution) pairs will be automatically generated during training and validation.
- (2) `1_SRResNet/SRResNet_test.py`
  - This source code is for testing the trained SRResNet model.
    - **PSNR**, **SSIM**
  - How to run?
    - E.g.
      - python SRResNet_test.py --gpu_index=0 --HR_dir='../Dataset/Set5/GTmod12' --LR_dir='../Dataset/Set5/LRbicx4' --generator_pt_path='MSE+Res34_3.pt'
    - Please see `1_SRResNet/SRResNet_parser.py` - `add_default_arguments()` & `add_arguments_for_test()` for all possible arguments to run the code.
    - Note that you have to prepare both **HR** and its corresponding **LR** dataset (x4). Please use the code `Dataset/high_resolution_to_low_resolution.py` for generating **LR** dataset if necessary.
- (3) `1_SRResNet/SRResNet_demo.py`
  - This source code is for generating and storing the super-resolution (x4) result of the LR input image.
  - How to run?
    - E.g.
      - python SRResNet_demo.py --gpu_index=0 --input_img_path='../Dataset/Set5/LRbicx4/baby.png' --output_img_path='baby_4x_MSE+Res34_3.png' --generator_pt_path='MSE+Res34_3.pt'
    - Please see `1_SRResNet/SRResNet_parser.py` - `add_default_arguments()` & `add_arguments_for_demo()` for all possible arguments to run the code.

### SRGAN
- (1) `2_SRGAN/SRGAN_train.py`
  - This source code is for training a SRGAN model.
  - How to run?
    - E.g.
      - python SRGAN_train.py --gpu_index=0 --batch_size=40 --tag='Example-SRGAN' --generator_pt_path='../1_SRResNet/MSE+Res34_3.pt' --train_dir='../Dataset/DIV2K/HR_dummy/train' --valid_dir='../Dataset/DIV2K/HR_dummy/valid' --epochs=100
    - Please see `2_SRGAN/SRGAN_parser.py` - `add_default_arguments()` & `add_arguments_for_train()` for all possible arguments to run the code.
    - Note that you only need to prepare **HR (High-Resolution) dataset**. LR (Low-Resolution) pairs will be automatically generated during training and validation.
    - You need to provide the pre-trained weight of the generator model (i.e. SRResNet). (Please train your SRResNet model first, before training SRGAN!)
- (2) `2_SRGAN/SRGAN_test.py`
  - This source code is for testing the trained SRGAN model.
    - **PSNR**, **SSIM**
  - How to run?
    - E.g.
      - python SRGAN_test.py --gpu_index=0 --HR_dir='../Dataset/Set5/GTmod12' --LR_dir='../Dataset/Set5/LRbicx4' --generator_pt_path='MSE+Res34_3_SRGAN.pt'
    - Please see `2_SRGAN/SRGAN_parser.py` - `add_default_arguments()` & `add_arguments_for_test()` for all possible arguments to run the code.
    - Note that you have to prepare both **HR** and its corresponding **LR** dataset (x4). Please use the code `Dataset/high_resolution_to_low_resolution.py` for generating **LR** dataset if necessary.
- (3) `2_SRGAN/SRGAN_demo.py`
  - This source code is for generating and storing the super-resolution (x4) result of the LR input image.
  - How to run?
    - E.g.
      - python SRGAN_demo.py --gpu_index=0 --input_img_path='../Dataset/Set5/LRbicx4/baby.png' --output_img_path='baby_4x_MSE+Res34_3.png' --generator_pt_path='MSE+Res34_3.pt'
    - Please see `2_SRGAN/SRGAN_parser.py` - `add_default_arguments()` & `add_arguments_for_demo()` for all possible arguments to run the code.

### Contribution
- If you find any bugs or have opinions for further improvements, feel free to contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.

### Reference
- For code design
  1. https://github.com/unique-chan/Knowledge-Distillation (My own github repo)
  2. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html (Custom data loader)
  3. https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python (PSNR, SSIM) ➜ I copied the whole source code from this repo.
  4. https://github.com/jorge-pessoa/pytorch-msssim (SSIM for PyTorch) ➜ I copied the whole source code from this repo.


- For debugging
  1. https://newbedev.com/pytorch-set-grad-enabled-false-vs-with-no-grad
  2. https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
