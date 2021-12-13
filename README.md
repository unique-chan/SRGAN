# SRGAN
## Unofficial Implementation for "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Model"
Yechan Kim (20212047)

### Prerequisites
- See `requirements.txt` for details.
~~~ME
torch
torchvision
matplotlib
opencv-contrib-python ??
numpy
tqdm
tensorboard
~~~

### How to Run (1): SRResNet
- blah blah

### How to Run (2): SRGAN
- blah blah

### Contribution
- If you find any bugs or have opinions for further improvements, feel free to contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.

### Reference
- For code design
  1. https://github.com/unique-chan/Knowledge-Distillation (My own github repo)
  2. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html (Custom data loader)
  3. https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python (PSNR, SSIM) ➜ I copied the whole source code from this repo.
  4. https://github.com/jorge-pessoa/pytorch-msssim (SSIM for PyTorch) ➜ I copied the whole source code from this repo.
- For debugging
  4. https://newbedev.com/pytorch-set-grad-enabled-false-vs-with-no-grad
  5. https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method