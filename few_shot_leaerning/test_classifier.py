import argparse
import os
import sys
from pathlib import Path

parent_path = Path(__file__).absolute().parent
sys.path.append(os.path.abspath(parent_path))
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from init_env import init_env


def main(config):
    #### Dataset ####
    # val
    eval_val = True
    val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
    val_loader = DataLoader(val_dataset, config['batch_size'], num_workers=8, pin_memory=True)
    utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))
    
    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True
        
        fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(
                fs_dataset[0][0].shape, len(fs_dataset),
                fs_dataset.n_classes))
        
        n_way = 5
        n_query = 15
        n_shots = [1, 5]
        fs_loaders = []
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(fs_dataset.label, 200, n_way, n_shot + n_query, ep_per_batch=4)
            fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler, num_workers=8, pin_memory=True)
            fs_loaders.append(fs_loader)
    else:
        eval_fs = False
    
    ########
    
    #### Model and Optimizer ####
    
    model_sv = torch.load(args.model_path)
    model = models.load(model_sv)
    
    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder
    
    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)
    
    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    ########
    
    aves_keys = ['tl', 'ta', 'vl', 'va']
    
    if eval_fs:
        for n_shot in n_shots:
            aves_keys += ['fsa-' + str(n_shot)]
    aves = {k: utils.Averager() for k in aves_keys}
    
    # eval
    if eval_val:
        model.eval()
        for data, label in tqdm(val_loader, desc='val', leave=False):
            data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)
            
            aves['vl'].add(loss.item())
            aves['va'].add(acc)
    
    if eval_fs:
        fs_model.eval()
        for i, n_shot in enumerate(n_shots):
            np.random.seed(0)
            for data, _ in tqdm(fs_loaders[i], desc='fs-' + str(n_shot), leave=False):
                x_shot, x_query = fs.split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=4)
                label = fs.make_nk_label(n_way, n_query, ep_per_batch=4).cuda()
                with torch.no_grad():
                    logits = fs_model(x_shot, x_query).view(-1, n_way)
                    acc = utils.compute_acc(logits, label)
                aves['fsa-' + str(n_shot)].add(acc)
    
    for k, v in aves.items():
        aves[k] = v.item()
    
    log_str = ''
    
    if eval_val:
        log_str += 'val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
    
    if eval_fs:
        log_str += ', fs'
        for n_shot in n_shots:
            key = 'fsa-' + str(n_shot)
            log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
    
    print(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--res_type', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    init_env()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    
    utils.set_gpu(args.gpu)
    main(config)
