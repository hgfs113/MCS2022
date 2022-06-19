import torch
from torch import nn
from torchvision import models
import timm


class IdentityWithInFeatures(nn.Module):
    def __init__(self, in_features, *args, **kwargs):
        super(IdentityWithInFeatures, self).__init__()
        self.in_features = in_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    if arch.startswith('resnet'):
        # allowed ARCH:
        #     resnet18 (12M)
        #     resnet34 (22M)
        #     resnet50 (26M)
        #     resnet101 (45M)
        #     resnet152 (60M)
        model = models.__dict__[arch](pretrained=True)
    elif arch.startswith('regnet'):
        # https://pytorch.org/vision/stable/models.html#id55
        # allowed ARCH:
        #     regnet_[x,y]_<N>
        model = models.__dict__[arch](pretrained=True)
    elif arch.startswith('seresnext'):
        # https://arxiv.org/pdf/1709.01507v4.pdf
        # allowed ARCH:
        #     seresnext26d_32x4d (17M)
        #     seresnext50_32x4d (28M)
        model = timm.create_model(arch, pretrained=True)
    else:
        raise Exception('model type is not supported:', arch)
    if config.train.loss.loss_type == 'cross_entropy':
        model.fc = nn.Linear(model.fc.in_features, config.dataset.num_of_classes)
    else:
        # pure embedding instead of classification head
        # for ArcFace and CosFace losses
        model.fc = IdentityWithInFeatures(model.fc.in_features)
    model.to('cuda')
    return model
