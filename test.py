
from utils import get_config, get_all_data_loaders
from trainer import Trainer
import argparse
from torch.autograd import Variable

import torch
import os
import numpy as np
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='config.yaml',
                    help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str,
                    help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int,
                    help="1 for a2b and others for b2a", default=1)
parser.add_argument('--num_style', type=int, default=10,
                    help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true',
                    help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true',
                    help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.',
                    help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

opts = parser.parse_args()

config = get_config(opts.config)
input_dim = config['input_dim']

# Setup model and data loader
data_loader = get_all_data_loaders(
    config)

trainer = Trainer(config)

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.eval()
basis_encode = trainer.gen_a.encode
trait_encode = trainer.gen_b.encode

abnormal_decode = trainer.gen_b.decode
normal_decode = trainer.gen_a.decode
config['batch_size'] = 1

(
    train_loader_a,
    train_loader_b,
    test_loader_a,
    test_loader_b) = get_all_data_loaders(
    config)


for _p in trainer.gen_a.trait_encoder.parameters():
    print(_p.data)


abnormals = []
normals = []
_n = 0
for vec_a in test_loader_a:
    for vec_b in test_loader_b:
        basis, _ = basis_encode(vec_a.float())
        _, trait = trait_encode(vec_b.float())
        output = abnormal_decode(basis, trait)
        abnormals.append(output.data.numpy())
        basis, _ = basis_encode(vec_b.float())
        _, trait = trait_encode(vec_a.float())
        output = normal_decode(basis, trait)
        # print(output.data.numpy())
        normals.append(output.data.numpy())
        _n += 1
        print(_n)


np.save('abnormals', abnormals)
np.save('normals', normals)
