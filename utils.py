import os
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_metric_learning import losses


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def save_checkpoint(model, criterion, optimizer, scheduler, epoch, outdir):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}.pth".format(epoch)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('loss', criterion.state_dict()),
        ('optimizer', optimizer.state_dict()),
        ('scheduler', scheduler.state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)


def load_checkpoint(model, criterion, optimizer, scheduler, filename):
    """Loads checkpoint from disk"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    criterion.load_state_dict(checkpoint['loss'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_optimizer(config, net):
    lr = config.train.learning_rate

    print("Opt: ", config.train.optimizer)

    if config.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=config.train.momentum,
                                    weight_decay=config.train.weight_decay)
    else:
        raise Exception("Unknown type of optimizer: {}".format(config.train.optimizer))
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.n_epoch)
    elif config.train.lr_schedule.name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.train.lr_schedule.step_size,
                                                    gamma=config.train.lr_schedule.gamma)
    else:
        raise Exception("Unknown type of lr schedule: {}".format(config.train.lr_schedule))
    return scheduler

def get_criterion(config, net):
    if config.train.loss.loss_type == 'cross_entropy':
        if not config.train.loss.label_smoothing:
            return torch.nn.CrossEntropyLoss()
        else:
            return LabelSmoothingCrossEntropy()
    elif config.train.loss.loss_type == 'arcface':
        # see https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss
        return losses.ArcFaceLoss(
            config.dataset.num_of_classes,
            net.fc.in_features,
            margin=config.train.loss.margin,
            scale=config.train.loss.scale,
        )
    elif config.train.loss.loss_type == 'cosface':
        # see https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#cosfaceloss
        return losses.CosFaceLoss(
            config.dataset.num_of_classes,
            net.fc.in_features,
            margin=config.train.loss.margin,
            scale=config.train.loss.scale,
        )
    else:
        raise NotImplementedError


def get_training_parameters(config, net):
    criterion = get_criterion(config, net).to('cuda')
    optimizer = get_optimizer(config, net)
    scheduler = get_scheduler(config, optimizer)
    return criterion, optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]


def calculate_embeddings_dist(embedding1, embedding2, distance='euclidean'):
    if distance == 'euclidean':
        return np.linalg.norm(embedding1 - embedding2)
    elif distance == 'cosine':
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    else:
        raise Exception('distance type is not supported:', distance)
