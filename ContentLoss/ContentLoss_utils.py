import torch.nn as nn


def get_extractor(torch_model, i=5, j=4):
    ''' [Logic]
    extract the feature extractor from torch_model
    here, the extractor should include from 1-st layer to the 'j'-th convolutional layer
          before 'i'-th max-pooling layer.
    '''
    layers = list(torch_model.children())[0]
    last_conv_idx_list = []
    last_maxpool_cnt = 0
    for idx, layer in enumerate(layers):
        if last_maxpool_cnt == i:
            break
        if 'MaxPool2d' in str(layer):
            last_maxpool_cnt += 1
        elif 'Conv2d' in str(layer):
            last_conv_idx_list.append(idx + 1)
    final_layers = layers[0:last_conv_idx_list[-(j + 1)]]
    return nn.Sequential(*final_layers)
