
from utils import get_config, get_all_data_loaders, prepare_sub_folder, write_loss
import argparse
from torch.autograd import Variable
from trainer import Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='fqmall.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

# Setup model and data loader
trainer = Trainer(config)
# trainer.cuda()
(
    train_loader_a,
    train_loader_b,
    test_loader_a,
    test_loader_b) = get_all_data_loaders(
    config)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(
    os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Start training
iterations = trainer.resume(
    checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    it = 0
    for data_a in test_loader_a:
        for data_b in test_loader_b:
            trainer.update_learning_rate()
            data_a, data_b = Variable(
                data_a).float(), Variable(data_b).float()
            # Main training code
            trainer.dis_update(data_a, data_b, config)
            trainer.gen_update(data_a, data_b, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            it += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
