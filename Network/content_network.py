import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo as model_zoo

class content_model(nn.Module):

    def __init__(self):
        super(content_model, self).__init__()
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"))
        self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

    def forward(self, x):
        return self.feature(x)