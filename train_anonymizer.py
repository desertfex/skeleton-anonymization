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

from utils import import_class, str2bool
from multiprocessing import Pool


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--anonymizer-model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--action-model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--action-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--privacy-model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--privacy-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--test-action',
        type=dict,
        default=dict())
    parser.add_argument(
        '--test-privacy',
        type=dict,
        default=dict())
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    parser.add_argument(
        '--anonymizer-base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--privacy-base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')

    parser.add_argument(
        '--anonymizer-steps',
        type=int,
        default=[0, 1, 2],
        nargs='+',
        help='steps to train anonymizer')
    parser.add_argument(
        '--privacy-steps',
        type=int,
        default=[4],
        nargs='+',
        help='steps to train privacy classifier')

    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
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
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')
    parser.add_argument(
        '--alpha',
        type=float,)
    parser.add_argument(
        '--beta',
        type=float,)
    parser.add_argument(
        '--pretrained-action',
        type=str,)
    parser.add_argument(
        '--pretrained-privacy',
        type=str,)
    parser.add_argument(
        '--pretrained-privacy-test',
        nargs='+',
        type=str,)
    parser.add_argument(
        '--test-feeder',
        type=str,)
    parser.add_argument(
        '--pretrained-anonymizer',
        default=None)

    return parser

# 0.5, 0.5 => maximum entorpy


def entropy(output):
    probs = torch.softmax(output, 1)
    log_probs = torch.log_softmax(output, 1)
    entropies = -torch.sum(probs * log_probs, 1)
    return torch.mean(entropies)


def reconsturction_loss(output, target):
    return nn.MSELoss()(output, target)


def action_classification_loss(output, target):
    action_classification_loss = nn.CrossEntropyLoss()(output, target)
    return action_classification_loss


def privacy_classification_loss(output, target):
    privacy_loss = nn.CrossEntropyLoss()(output, target)
    return privacy_loss


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.visualization_queue = []
        self.visualization_pool = Pool()
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

        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)
                self.train_writer = SummaryWriter(
                    os.path.join(logdir, 'train'), 'train')
                self.val_privacy_writers = [SummaryWriter(
                    os.path.join(logdir, f'val_privacy_{i}'), f'val_privacy_{i}') for i in range(len(arg.pretrained_privacy_test))]
                self.val_action_writer = SummaryWriter(
                    os.path.join(logdir, 'val_action'), 'val_action')
            else:
                self.train_writer = SummaryWriter(
                    os.path.join(logdir, 'debug'), 'debug')
                self.val_privacy_writers = [SummaryWriter(
                    os.path.join(logdir, f'debug_privacy_{i}'), f'debug_privacy_{i}') for i in range(len(arg.pretrained_privacy_test))]
                self.val_action_writer = SummaryWriter(
                    os.path.join(logdir, 'debug_val_action'), 'debug_val_action')

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        self.global_step = 0

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(
                    f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.anonymizer = nn.DataParallel(
                    self.anonymizer,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )
                self.action_classifier = nn.DataParallel(
                    self.action_classifier,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )
                self.privacy_classifier = nn.DataParallel(
                    self.privacy_classifier,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_eval_action_model(self, weight):
        self.print_log("Using action weight %s" % weight)
        self.eval_action_model = import_class(self.arg.action_model)(
            **self.arg.action_model_args).cuda(self.output_device)
        self.eval_action_model.load_state_dict(torch.load(weight))

        self.eval_action_model = nn.DataParallel(
            self.eval_action_model,
            device_ids=self.arg.device,
            output_device=self.output_device
        )
        self.eval_action_model.eval()

    def load_eval_privacy_models(self, weights):
        self.eval_privacy_models = []
        for weight in weights:
            self.print_log("Using privacy weight %s" % weight)
            self.eval_privacy_models.append(import_class(self.arg.privacy_model)(
                **self.arg.privacy_model_args).cuda(self.output_device))

            self.eval_privacy_models[-1].load_state_dict(torch.load(weight))
            self.eval_privacy_models[-1] = nn.DataParallel(
                self.eval_privacy_models[-1],
                device_ids=self.arg.device,
                output_device=self.output_device
            )
            self.eval_privacy_models[-1].eval()

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        AnonymizerModel = import_class(self.arg.anonymizer_model)

        # Copy model file and main
        shutil.copy2(inspect.getfile(AnonymizerModel), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        print(self.arg.model_args)

        self.anonymizer = AnonymizerModel(
            **self.arg.model_args).cuda(output_device)

        self.action_classifier = import_class(self.arg.action_model)(
            **self.arg.action_model_args).cuda(self.output_device)

        self.privacy_classifier = import_class(self.arg.privacy_model)(
            **self.arg.privacy_model_args).cuda(self.output_device)

        if self.arg.pretrained_action:
            self.print_log("Using pretrained action model %s" %
                           self.arg.pretrained_action)
            self.action_classifier.load_state_dict(
                torch.load(self.arg.pretrained_action))

        if self.arg.pretrained_privacy:
            self.print_log("Using pretrained privacy model %s" %
                           self.arg.pretrained_privacy)
            self.privacy_classifier.load_state_dict(
                torch.load(self.arg.pretrained_privacy))

        self.action_classifier.eval()

        if self.arg.pretrained_anonymizer:
            self.print_log(
                f'Loading weights from {self.arg.pretrained_anonymizer}')
            if '.pkl' in self.arg.pretrained_anonymizer:
                with open(self.arg.pretrained_anonymizer, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.pretrained_anonymizer)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.anonymizer.load_state_dict(weights)
            except:
                state = self.anonymizer.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.anonymizer.load_state_dict(state)
            self.anonymizer.fix_encoder()

        self.print_log("Loading models for evaluation")
        self.load_eval_action_model(self.arg.pretrained_action)
        self.load_eval_privacy_models(self.arg.pretrained_privacy_test)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.anonymizer.named_parameters():
            self.param_groups['anonymizer'].append(params)
        for name, params in self.action_classifier.named_parameters():
            self.param_groups['action_classifier'].append(params)
        for name, params in self.privacy_classifier.named_parameters():
            self.param_groups['privacy_classifier'].append(params)

        self.optim_param_groups = {
            'privacy_classifier': {'params': self.param_groups['privacy_classifier']},
            'action_classifier': {'params': self.param_groups['action_classifier']},
            'anonymizer': {'params': self.param_groups['anonymizer']}
        }

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.anonymizer_optimizer = optim.SGD(
                [self.optim_param_groups['anonymizer']],
                lr=self.arg.anonymizer_base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            self.privacy_classifier_optimizer = optim.SGD(
                [self.optim_param_groups['privacy_classifier']],
                lr=self.arg.privacy_base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.anonymizer_optimizer = optim.Adam(
                [self.optim_param_groups['anonymizer']],
                lr=self.arg.anonymizer_base_lr,
                weight_decay=self.arg.weight_decay)
            self.privacy_classifier_optimizer = optim.Adam(
                [self.optim_param_groups['privacy_classifier']],
                lr=self.arg.privacy_base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(
                'Unsupported optimizer: {}'.format(self.arg.optimizer))

    def load_lr_scheduler(self):
        self.privacy_classifier_lr_scheduler = MultiStepLR(
            self.privacy_classifier_optimizer, milestones=self.arg.step, gamma=0.1)
        self.anonymizer_lr_scheduler = MultiStepLR(
            self.anonymizer_optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)[
                'privacy_classfier_lr_scheduler_states']
            self.print_log(
                f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.privacy_classifier_lr_scheduler.load_state_dict(torch.load(
                self.arg.checkpoint)['privacy_classfier_lr_scheduler_states'])
            self.anonymizer_lr_scheduler.load_state_dict(torch.load(self.arg.checkpoint)[
                                                         'anonymizer_lr_scheduler_states'])
            # self.action_classifier_lr_scheduler.load_state_dict(torch.load(
            #     self.arg.checkpoint)['action_classifier_lr_scheduler_states'])
            self.print_log(
                f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(
                f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'privacy_classfication_optimizer_states': self.privacy_classifier_optimizer.state_dict(),
            'anonymization_optimizer_states': self.anonymizer_optimizer.state_dict(),
            'privacy_classfier_lr_scheduler_states': self.privacy_classifier_lr_scheduler.state_dict(),
            'anonymizer_lr_scheduler_states': self.anonymizer_lr_scheduler.state_dict(),
            # 'action_classifier_lr_scheduler_states': self.action_classifier_lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-bs{self.arg.batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, model, name, out_folder):
        state_dict = model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'{name}-weights-{epoch}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    # minimization step
    def anonymizer_step(self, data, gender_label, action_label,
                        recon_loss_list, action_loss_list, privacy_loss_list,
                        action_acc_list, privacy_acc_list, total_loss_list, timer):
        self.anonymizer_optimizer.zero_grad()
        anonymized = self.anonymizer(data)

        action = self.action_classifier(anonymized)
        action_loss = action_classification_loss(action, action_label)
        _, predict_action = torch.max(action, 1)
        action_acc = torch.mean((predict_action == action_label).float())

        gender = self.privacy_classifier(anonymized)
        privacy_loss = entropy(gender)
        _, predict_gender = torch.max(gender, 1)
        privacy_acc = torch.mean((predict_gender ==
                                  gender_label).float()).item()

        recon_loss = reconsturction_loss(anonymized, data)

        anonymization_loss = recon_loss - \
            self.arg.alpha * privacy_loss + self.arg.beta * action_loss

        anonymization_loss.backward()
        self.anonymizer_optimizer.step()
        timer['model'] += self.split_time()

        recon_loss_list.append(recon_loss.item())
        total_loss_list.append(anonymization_loss.item())
        action_loss_list.append(action_loss.item())
        privacy_loss_list.append(privacy_loss.item())

        action_acc_list.append(action_acc.item())
        privacy_acc_list.append(privacy_acc)

        self.train_writer.add_scalar(
            'action_acc', action_acc, self.global_step)
        self.train_writer.add_scalar(
            'recon_loss', recon_loss, self.global_step)
        self.train_writer.add_scalar(
            'action_loss', action_loss, self.global_step)
        self.train_writer.add_scalar(
            'privacy_loss', privacy_loss, self.global_step)
        self.train_writer.add_scalar(
            'privacy_acc', privacy_acc, self.global_step)
        self.train_writer.add_scalar(
            'total_loss', anonymization_loss, self.global_step)
        lr = self.anonymizer_optimizer.param_groups[0]['lr']
        self.train_writer.add_scalar('anonymizer_lr', lr, self.global_step)
        timer['statistics'] += self.split_time()

    # maximization step
    def privacy_classifier_step(self, data, gender_label, privacy_loss_list, privacy_acc_list, timer):
        self.privacy_classifier_optimizer.zero_grad()

        anonymized = self.anonymizer(data)
        gender = self.privacy_classifier(anonymized)
        privacy_loss = privacy_classification_loss(gender, gender_label)
        _, predict_gender = torch.max(gender, 1)
        privacy_acc = torch.mean(
            (predict_gender == gender_label).float()).item()
        privacy_loss.backward()
        self.privacy_classifier_optimizer.step()

        timer['model'] += self.split_time()

        # privacy_loss_list.append(privacy_loss.item())
        privacy_acc_list.append(privacy_acc)
        self.train_writer.add_scalar(
            'privacy_acc', privacy_acc, self.global_step)
        # self.train_writer.add_scalar(
        #     'privacy_loss', privacy_loss, self.global_step)
        lr = self.privacy_classifier_optimizer.param_groups[0]['lr']
        self.train_writer.add_scalar(
            'privacy_classifier_lr', lr, self.global_step)
        timer['statistics'] += self.split_time()

    def train(self, epoch, save_model=False):
        self.anonymizer.train()
        loader = self.data_loader['train']
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.anonymizer_optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)
        total_loss_list = []
        recon_loss_list = []
        action_loss_list = []
        privacy_loss_list = []
        privacy_acc_list = []
        action_acc_list = []

        if len(self.arg.privacy_steps) == 0 or len(self.arg.anonymizer_steps) == 0:
            rotation_step = 1
        else:
            rotation_step = max(max(self.arg.anonymizer_steps),
                                max(self.arg.privacy_steps)) + 1

        for batch_idx, (data, gender_label, action_label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                gender_label = gender_label.long().cuda(self.output_device)
                action_label = action_label.long().cuda(self.output_device)

            timer['dataloader'] += self.split_time()

            if (self.global_step % rotation_step) in self.arg.privacy_steps:  # privacy step
                self.privacy_classifier_step(
                    data, gender_label, privacy_loss_list, privacy_acc_list, timer)
            else:  # anonymizer step
                self.anonymizer_step(data, gender_label, action_label, recon_loss_list,
                                     action_loss_list, privacy_loss_list, action_acc_list, privacy_acc_list, total_loss_list, timer)

            mean_recon_loss = np.mean(recon_loss_list) if len(
                recon_loss_list) else np.nan
            mean_action_loss = np.mean(action_loss_list) if len(
                action_loss_list) else np.nan
            mean_action_acc = np.mean(action_acc_list) if len(
                action_acc_list) else np.nan
            mean_privacy_loss = np.mean(privacy_loss_list) if len(
                privacy_loss_list) else np.nan
            mean_privacy_acc = np.mean(privacy_acc_list) if len(
                privacy_acc_list) else np.nan

            process.set_description(
                '(BS %d) recon_loss: %.4f, action_loss: %.4f, priv_loss: %.4f, action_acc: %.4f, priv_acc: %.4f' % (
                    self.arg.batch_size, mean_recon_loss, mean_action_loss,
                    mean_privacy_loss, mean_action_acc, mean_privacy_acc
                )
            )

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(total_loss_list)
        self.print_log(
            f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss:.4f}).')
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        self.anonymizer_lr_scheduler.step()
        # self.action_classifier_lr_scheduler.step()
        # self.privacy_classifier_lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch, self.anonymizer,
                              "anonymizer", 'anonymizer_weights')
            self.save_checkpoint(epoch)

    def top_k(self, score, labels, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:]
                     for i, l in enumerate(labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def eval_action_validate(self, weight):
        # test eval model
        self.print_log(f'Action: eval test')
        loss_values = []
        action_batches = []
        labels = []
        self.eval_action_model.eval()
        with torch.no_grad():
            process = tqdm(self.test_loader_action, dynamic_ncols=True)
            for batch_idx, (original, anonymized, gender_label, action_label, index) in enumerate(process):
                labels.extend(action_label.cpu().tolist())
                action_label = action_label.long().cuda(self.output_device)
                action = self.eval_action_model(anonymized)
                loss = action_classification_loss(action, action_label)
                loss_values.append(loss.item())
                action_batches.append(action.data.cpu().numpy())
        score = np.concatenate(action_batches)
        loss = np.mean(loss_values)
        accuracy = self.top_k(score, labels, 1)
        self.print_log(f'Test action {weight}, loss: {loss}')
        self.print_log(f'Test action {weight}, acc: {accuracy}')
        self.print_log(
            f'\tMean test loss of {len(self.test_loader_action)} batches: {np.mean(loss_values)}.')
        for k in self.arg.show_topk:
            self.print_log(
                f'\tTop {k}: {100 * self.top_k(score, labels, k):.2f}%')

        self.val_action_writer.add_scalar(
            'action_acc', accuracy, self.global_step)
        self.val_action_writer.add_scalar(
            'action_loss', loss, self.global_step)
        self.val_action_writer.add_scalar(
            'recon_loss', self.test_feeder.get_reconstruction_loss(), self.global_step)

    def eval_privacy_validate(self, weight, privacy_idx):
        # test eval model
        self.print_log(f'Privacy: eval test')
        loss_values = []
        gender_batches = []
        labels = []
        model = self.eval_privacy_models[privacy_idx]
        model.eval()
        with torch.no_grad():
            process = tqdm(self.test_loader_privacy, dynamic_ncols=True)
            for batch_idx, (original, anonymized, gender_label, action_label, index) in enumerate(process):
                labels.extend(gender_label.cpu().tolist())
                gender_label = gender_label.long().cuda(self.output_device)
                gender = model(anonymized)
                loss = entropy(gender)  # , gender_label)
                loss_values.append(loss.item())
                gender_batches.append(gender.data.cpu().numpy())
        score = np.concatenate(gender_batches)
        loss = np.mean(loss_values)
        accuracy = self.top_k(score, labels, 1)
        self.print_log(f'Test {weight}, loss: {loss}')
        self.print_log(f'Test {weight}, acc: {accuracy}')
        self.print_log(
            f'\tMean test loss of {len(self.test_loader_privacy)} batches: {np.mean(loss_values)}.')
        for k in self.arg.show_topk:
            self.print_log(
                f'\tTop {k}: {100 * self.top_k(score, labels, k):.2f}%')
        print(privacy_idx, accuracy, self.global_step)
        self.val_privacy_writers[privacy_idx].add_scalar(
            'privacy_acc', accuracy, self.global_step)
        self.val_privacy_writers[privacy_idx].add_scalar(
            'privacy_loss', loss, self.global_step)

    def build_test_feeder(self):
        TestFeeder = import_class(self.arg.test_feeder)
        self.test_feeder = TestFeeder(**self.arg.test_feeder_args,
                                      anonymizer=self.anonymizer,
                                      output_device=self.output_device)

    def build_test_loader(self):
        self.build_test_feeder()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        self.test_loader_action = torch.utils.data.DataLoader(
            dataset=self.test_feeder,
            batch_size=self.arg.test_action['batch_size'],
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)
        self.test_loader_privacy = torch.utils.data.DataLoader(
            dataset=self.test_feeder,
            batch_size=self.arg.test_privacy['batch_size'],
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def visualize(self, epoch):
        visualization_path = os.path.join(
            self.arg.work_dir, 'visualizations', 'epoch_%d' % epoch)
        os.makedirs(visualization_path, exist_ok=True)
        res = self.visualization_fn(
            data_path=self.arg.test_feeder_args['data_path'],
            action_label_path=self.arg.test_feeder_args['action_label_path'],
            gender_label_path=self.arg.test_feeder_args['gender_label_path'],
            output_path=visualization_path,
            filenames=self.visualization_filenames,
            anonymizer=self.anonymizer,
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

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size
            self.visualize_original()

            # pre-validation with default parameters
            self.build_test_loader()
            self.eval_action_validate(self.arg.pretrained_action)
            for i, weight in enumerate(self.arg.pretrained_privacy_test):
                self.eval_privacy_validate(weight, i)

            for epoch in range(self.arg.start_epoch + 1, self.arg.num_epoch + 1):
                self.train(epoch, save_model=True)
                self.visualize(epoch)
                self.build_test_loader()
                self.eval_action_validate(self.arg.pretrained_action)
                for i, weight in enumerate(self.arg.pretrained_privacy_test):
                    self.eval_privacy_validate(weight, i)
                self.check_visualization_progress()
                if self.arg.debug and epoch > 10:
                    break

            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(
                f'Anonymizer Base LR: {self.arg.anonymizer_base_lr}')
            # self.print_log(
            #     f'Privacy Base LR: {self.arg.privacy_base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.wait_remaining_visualization()


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
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
