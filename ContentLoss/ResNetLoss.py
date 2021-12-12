import torch
import torch.nn as nn


def get_extractor_for_ResNet(torch_model, i=6):
    ''' [Logic]
    extract the feature extractor from torch_model (ResNet-model)
    here, the extractor should include from 1-st layer to 'i'-th residual block.
    '''
    layers = list(torch_model.children())
    pre_layers = []
    post_blocks = []

    # for pre_layers
    for layer in layers:
        if type(layer) == nn.Sequential:
            break
        pre_layers.append(layer)
    # for post_blocks
    cnt = 0
    for sequence in layers[5:]:
        if type(sequence) == nn.Sequential:
            for block in sequence:
                post_blocks.append(block)
                cnt += 1
                if cnt > i:
                    break
    # freezing (1)
    for layer in pre_layers:
        layer.requires_grad = False
    # freezing (2)
    for block in post_blocks:
        block.requires_grad = False
    final_layers = pre_layers + post_blocks
    return nn.Sequential(*final_layers)


class ResNetLoss(nn.Module):
    def __init__(self, i=5, device='cuda:0', resnet_model_name='resnet34'):
        ''' [Logic]
        1) extract the feature map from the 'i'-th residual block
           for both hr_image (ground_truth) and sr_image (prediction).
        2) compute and return mse_loss between features maps of hr_image & sr_image.
        '''
        super(ResNetLoss, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', resnet_model_name, pretrained=True).to(device).eval()
        self.extractor = get_extractor_for_ResNet(resnet, i)
        self.mse_loss = nn.MSELoss()

    def forward(self, hr_img, sr_img):
        hr_feature_map = self.extractor(hr_img)
        sr_feature_map = self.extractor(sr_img)
        vgg_loss = self.mse_loss(hr_feature_map, sr_feature_map)
        return vgg_loss
