#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from model.ae_anonymizer import Anonymizer

from utils import import_class, str2bool
from feeders.feeder_anonymization import Feeder
from multiprocessing import Pool


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')

    return parser


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def reconsturction_loss(output, target):
    return nn.MSELoss()(output, target)


class Processor():
    def __init__(self, arg):

        self.arg = arg
        self.save_arg()
        self.output_device = arg.device[0] if type(
            arg.device) is list else arg.device
        self.global_step = 0
        self.num_epoch = arg.num_epoch
        self.work_dir = arg.work_dir
        self.batch_size = arg.batch_size
        if "etri" in arg.train_feeder_args['data_path']:
            self.data_type = "ETRI"
        else:
            self.data_type = "NTU"
        if self.data_type == "NTU":
            from ntu_visualize import visualize
            self.visualization_fn = visualize
            self.visualization_filenames = [
                'S001C001P002R001A024.skeleton',
                'S011C001P027R002A023.skeleton',
                'S014C001P007R001A012.skeleton',
                'S017C001P020R001A035.skeleton'
            ]
        else:
            from etri_visualize import visualize
            self.visualization_fn = visualize
            self.visualization_filenames = [
                'A001_P045_G001_C001.csv',
                'A001_P054_G001_C001.csv',
                'A005_P045_G001_C001.csv',
                'A005_P054_G001_C001.csv',
                'A013_P045_G003_C007.csv',
                'A013_P054_G003_C007.csv'
            ]

        if self.data_type == "NTU":
            self.model = Anonymizer(
                num_point=25, num_person=2).cuda(self.output_device)
        else:
            self.model = Anonymizer(
                num_point=25, num_person=1).cuda(self.output_device)
        self.visualization_queue = []
        self.visualization_pool = Pool()
        if type(arg.device) is list:
            print("Use DataParallel")
            self.model = nn.DataParallel(
                self.model,
                device_ids=arg.device,
                output_device=self.output_device
            )

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(arg.seed + worker_id + 1)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=Feeder(**arg.train_feeder_args),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            drop_last=True,
            worker_init_fn=worker_seed_fn)

        param_group = []
        for name, params in self.model.named_parameters():
            param_group.append(params)

        # self.optimizer = optim.SGD(
        #     [{'params': param_group}],
        #     lr=arg.base_lr,
        #     momentum=0.9,
        #     nesterov=True,
        #     weight_decay=arg.weight_decay)

        self.optimizer = optim.Adam(
            [{'params': param_group}],
            lr=arg.base_lr,
            weight_decay=arg.weight_decay)
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=arg.step, gamma=0.1)

    def train_epoch(self, epoch):
        self.model.train()
        loss_list = []
        loader = self.train_loader
        process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, gender_label, action_label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
            self.optimizer.zero_grad()
            anonymized = self.model(data)
            loss = reconsturction_loss(anonymized, data)
            loss.backward()
            loss_list.append(loss.item())
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]['lr']
            process.set_description(
                '(Epoch: %d, BS %d, lr: %.2e) recon_loss: %.4f' % (
                    epoch, self.batch_size, lr, np.mean(loss_list))
            )
        self.lr_scheduler.step()

        # save training checkpoint & weights
        self.save_weights(epoch, self.model,
                          "model", "ae_pretrain_weights")

    def test_epoch(self, epoch):
        pass

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_weights(self, epoch, model, name, out_folder):
        state_dict = model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'{name}-weights-{epoch}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def visualize(self, epoch):
        # self.print_log('Visualization')
        visualization_path = os.path.join(
            self.arg.work_dir, 'visualizations', 'epoch_%d' % epoch)
        os.makedirs(visualization_path, exist_ok=True)
        res = self.visualization_fn(
            data_path=self.arg.test_feeder_args['data_path'],
            action_label_path=self.arg.test_feeder_args['action_label_path'],
            gender_label_path=self.arg.test_feeder_args['gender_label_path'],
            output_path=visualization_path,
            filenames=self.visualization_filenames,
            anonymizer=self.model,
            device=self.output_device,
            pool=self.visualization_pool)
        self.visualization_queue.append((epoch, res))

    def visualize_original(self):
        visualization_path = os.path.join(
            self.arg.work_dir, 'visualizations', 'original')
        os.makedirs(visualization_path, exist_ok=True)
        res = self.visualization_fn(
            data_path=self.arg.test_feeder_args['data_path'],
            action_label_path=self.arg.test_feeder_args['action_label_path'],
            gender_label_path=self.arg.test_feeder_args['gender_label_path'],
            output_path=visualization_path,
            filenames=self.visualization_filenames,
            anonymizer=None,
            device=self.output_device,
            pool=self.visualization_pool)
        self.visualization_queue.append((-1, res))

    def check_visualization_progress(self):
        new_queue = []
        done = []
        running = []
        for epoch, res in self.visualization_queue:
            if epoch == -1:
                epoch = "original"
            if res.ready():
                done.append(str(epoch))
            else:
                running.append(str(epoch))
                new_queue.append((epoch, res))
        if done:
            print("visualization %s are finished" % (", ".join(done)))
        if running:
            print("visualization %s are still running" % (", ".join(running)))
        self.visualization_queue = new_queue

    def wait_remaining_visualization(self):
        print('Waiting remaining visualization processes...')
        for epoch, res in self.visualization_queue:
            if epoch == -1:
                epoch = "original"
            res.wait()
            print("visualization %d is done" % epoch)
        self.visualization_queue = []
        self.visualization_pool.close()
        self.visualization_pool.join()

    def save_weights(self, epoch, model, name, out_folder):
        state_dict = model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'{name}-weights-{epoch}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
            print(s, file=f)

    def start(self):
        self.visualize_original()

        for epoch in range(1, self.num_epoch+1):
            # self.model.teacher_force = True
            self.train_epoch(epoch)
            # self.model.teacher_force = False
            self.visualize(epoch)
            self.test_epoch(epoch)
            self.check_visualization_progress()
        self.wait_remaining_visualization()


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()
