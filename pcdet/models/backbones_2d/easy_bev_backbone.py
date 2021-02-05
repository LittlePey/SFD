import numpy as np
import torch
import torch.nn as nn
from functools import partial


class EasyBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        num_filters = 160

        Conv2d = partial(nn.Conv2d, bias=False)
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        self.conv0 = Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn0 = BatchNorm2d(num_filters)

        self.conv1 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = BatchNorm2d(num_filters)

        self.conv2 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = BatchNorm2d(num_filters)

        self.conv3 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = BatchNorm2d(num_filters)

        self.conv4 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = BatchNorm2d(num_filters)

        self.conv5 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn5 = BatchNorm2d(num_filters)
        
        self.relu = nn.ReLU()
        
        self.num_bev_features = num_filters

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        data_dict['spatial_features_2d'] = x

        return data_dict
