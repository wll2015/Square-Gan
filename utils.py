
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from data import FeatureDataset
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (
            classname.find(
                'Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_all_data_loaders(conf):

    data_path = conf['data_path']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    trainA_loader = get_data_loader(
        'trainA.npy', data_path, batch_size, True, num_workers)
    trainB_loader = get_data_loader(
        'trainB.npy', data_path, batch_size, True, num_workers)
    testA_loader = get_data_loader(
        'testA.npy', data_path, batch_size, True, num_workers)
    testB_loader = get_data_loader(
        'testB.npy', data_path, batch_size, True, num_workers)

    return (trainA_loader, trainB_loader, testA_loader, testB_loader)


def get_data_loader(filename, data_path, batch_size, train, num_workers=4):

    dataset = FeatureDataset(filename, data_path)

    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=train, drop_last=True, num_workers=num_workers)

    return loader


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(
        getattr(
            trainer, attr
        )
    ) and not attr.startswith(
        "__") and (
        'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def prepare_sub_folder(output_directory):
    data_directory = os.path.join(output_directory, 'data')
    if not os.path.exists(data_directory):
        print("Creating directory: {}".format(data_directory))
        os.makedirs(data_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, data_directory


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler
