import torch
import torch.nn as nn
from torchvision.models import resnet18

# torch.manual_seed(0)


def get_model(device):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 3)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.to(device, dtype=torch.float)
    return model
