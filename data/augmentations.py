import torchvision as tv

normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        train_augs = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(config.dataset.input_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomRotation(45),
            tv.transforms.RandomPosterize(bits=2, p=0.3),
            tv.transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            tv.transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
            tv.transforms.RandomApply([
                tv.transforms.ColorJitter(
                    brightness=2, contrast=2,
                    saturation=2, hue=0.2),
            ], p=0.5),
            tv.transforms.ToTensor(),
            normalize
        ])
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return train_augs


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        val_augs = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(config.dataset.input_size),
            tv.transforms.ToTensor(),
            normalize
        ])
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return val_augs
