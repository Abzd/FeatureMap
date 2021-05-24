import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class CNN(nn.Module):
    def __init__(self, transform='NT'):
        # data dimension: [datetime, feature] = [10, 5]
        # NT: [5, 10]
        # GAF: [5, 10, 10]
        super(CNN, self).__init__()

        self.transform = transform
        if self.transform == 'NT':
            self.conv1 = nn.Conv2d(1, 1, (1, 2), stride=1, padding=0)
            self.conv2 = nn.Conv2d(1, 1, (5, 2), stride=1, padding=0)
            self.fc = nn.Linear(8, 2)
        elif self.transform == 'GAF':
            self.conv1 = nn.Conv2d(5, 3, (3, 3), stride=1, padding=0)
            self.max_pool1 = nn.MaxPool2d((2, 2), stride=1)
            self.conv2 = nn.Conv2d(3, 1, (3, 3), stride=2, padding=0)
            self.max_pool2 = nn.MaxPool2d((3, 3), stride=1)
            self.fc = nn.Linear(121, 2)

        else:
            raise ValueError('Not supported type.')

    def forward(self, x):
        x = self.conv1(x)

        if self.transform == 'GAF':
            x = self.max_pool1(x)
            x = self.conv2(x)
            x = self.max_pool2(x)
            x = x.view(x.size(0), 121)

        elif self.transform == 'NT':
            x = x.view(x.size(0), 8)

        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CNN('GAF')
    
    # print('---Layers---')
    # print(model.layers)

    print('---Model---')
    print(model)

