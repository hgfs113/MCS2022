import os
import sys
import yaml
import random
import argparse
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm

import utils
from models import models
from data import get_dataloader
from train import train, validation
from utils import convert_dict_to_tuple
from torch.utils.tensorboard import SummaryWriter


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    tensorboard_writer = SummaryWriter(args.logs)
    print(f"Tensorboard logs will be written in {args.logs}")

    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)

    print("Loading model...")
    net = models.load_model(config)
    criterion, optimizer, scheduler = utils.get_training_parameters(config, net)
    start_epoch = 0
    if args.checkpoint != '':
        print("Start from checkpoint:", args.checkpoint)
        start_epoch = utils.load_checkpoint(net, criterion, optimizer, scheduler, args.checkpoint)

    if config.num_gpu > 1:
        net = torch.nn.DataParallel(net)
    print("Done.")

    train_epoch = tqdm(range(start_epoch, config.train.n_epoch), dynamic_ncols=True, desc='Epochs', position=0)

    # main process
    best_eer = 1
    for epoch in train_epoch:
        train(net, train_loader, criterion, optimizer, config, epoch, tensorboard_writer)
        epoch_avg_acc, epoch_eer = validation(net, val_loader, criterion, epoch, tensorboard_writer)
        if epoch_eer <= best_eer:
            utils.save_checkpoint(net, criterion, optimizer, scheduler, epoch, outdir)
            best_eer = epoch_eer
        scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/baseline_mcs.yml', help='Path to config file.')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint.')
    parser.add_argument('--logs', type=str, default='log_tb', help='Path to logs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
