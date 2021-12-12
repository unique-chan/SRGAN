import torch
import torch.nn as nn


def get_extractor_for_VGG(torch_model, i=5, j=4):
    ''' [Logic]
    extract the feature extractor from torch_model (vgg-model)
    here, the extractor should include from 1-st layer to the 'j'-th convolutional layer
          before 'i'-th max-pooling layer.
    '''
    layers = list(torch_model.children())[0]
    last_conv_idx_list = []
    last_maxpool_cnt = 0
    for idx, layer in enumerate(layers):
        layer.requires_grad = False  # freezing!
        if last_maxpool_cnt == i:
            break
        if 'MaxPool2d' in str(layer):
            last_maxpool_cnt += 1
        elif 'Conv2d' in str(layer):
            last_conv_idx_list.append(idx + 1)
    final_layers = layers[0:last_conv_idx_list[-(j + 1)]]
    return nn.Sequential(*final_layers)


class VGGLoss(nn.Module):
    def __init__(self, i=5, j=4, device='cuda:0', vgg_model_name='vgg19_bn'):
        ''' [Logic]
        1) extract the feature map from the 'j'-th convolutional layer before 'i'-th max-pooling layer
           for both hr_image (ground_truth) and sr_image (prediction).
        2) compute and return mse_loss between features maps of hr_image & sr_image.
        '''
        super(VGGLoss, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', vgg_model_name, pretrained=True).to(device).eval()
        self.extractor = get_extractor_for_VGG(vgg, i, j)
        self.mse_loss = nn.MSELoss()

    def forward(self, hr_img, sr_img):
        hr_feature_map = self.extractor(hr_img)
        sr_feature_map = self.extractor(sr_img)
        vgg_loss = self.mse_loss(hr_feature_map, sr_feature_map)
        return vgg_loss
