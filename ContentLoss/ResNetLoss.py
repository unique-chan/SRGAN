import torch
import torch.nn as nn

from ContentLoss.ContentLoss_utils import get_extractor


class ResNetLoss(nn.Module):
    def __init__(self, i=5, j=4, device='cuda:0', resnet_model_name='resnet34'):
        ''' [Logic]
        1) extract the feature map from the 'j'-th convolutional layer before 'i'-th max-pooling layer
           for both hr_image (ground_truth) and sr_image (prediction).
        2) compute and return mse_loss between features maps of hr_image & sr_image.
        '''
        super(ResNetLoss, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', resnet_model_name, pretrained=True).to(device).eval()
        self.extractor = get_extractor(resnet, i, j)
        self.mse_loss = nn.MSELoss()

    def forward(self, hr_img, sr_img):
        with torch.no_grad():
            hr_feature_map = self.extractor(hr_img)
            sr_feature_map = self.extractor(sr_img)
            vgg_loss = self.mse_loss(hr_feature_map, sr_feature_map)
            return vgg_loss
