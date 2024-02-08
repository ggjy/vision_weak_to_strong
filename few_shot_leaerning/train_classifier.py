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
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from init_env import init_env
from utils.kd_utils import get_kd_loss, kd_epoch_init, init_class_teacher_model, init_classifier_kd_svname


def main(config):
    svname = args.name
    if svname is None:
        if config.get('teacher_load'):
            svname = init_classifier_kd_svname(config)
        else:
            svname = f"classifier_{config['train_dataset']}_{config['model_args']['encoder']}"
        
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    
    save_root = os.getenv('OUT_PATH', './save')
    save_path = os.path.join(save_root, svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    #### Dataset ####
    
    # train
    train_dataset = datasets.make(config['train_dataset'], **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape,
                                                   len(train_dataset), train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    
    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'], num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset),
                                                     val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False
    
    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True
        
        fs_dataset = datasets.make(config['fs_dataset'], **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(fs_dataset[0][0].shape, len(fs_dataset), fs_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)
        
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
    
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])
    
    if config.get('teacher_load'):
        teacher_model = init_class_teacher_model(config)
    
    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder
    
    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)
        if config.get('teacher_load'):
            teacher_model = nn.DataParallel(teacher_model)
    
    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    
    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
    
    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    
    for epoch in range(1, max_epoch + 1 + 1):
        start_iter = 0
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        if config.get('teacher_load'):
            aves_keys = kd_epoch_init(config, aves_keys)
        
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]
        aves = {k: utils.Averager() for k in aves_keys}
        
        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)
            
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            
            if config.get('teacher_load'):
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                loss, aves = get_kd_loss(config, logits, teacher_logits, epoch, max_epoch, aves, loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logits = None
            loss = None
        
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
        
        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
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
        
        # post
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        for k, v in aves.items():
            aves[k] = v.item()
        
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        
        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        if config.get('teacher_load'):
            log_str = (f"epoch {epoch_str}, lr {optimizer.param_groups[0]['lr']:.6f}, "
                       f"train {aves['tl']:.4f},{aves['tlkd']:.4f}|{aves['ta']:.4f}")
        else:
            log_str = 'epoch {}, lr {:.6f}, train {:.4f}|{:.4f}'.format(
                    epoch_str, optimizer.param_groups[0]['lr'], aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)
        
        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)
        
        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)
        
        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)
        
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model
        
        training = {
                'epoch': epoch,
                'optimizer': config['optimizer'],
                'optimizer_args': config['optimizer_args'],
                'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
                'file': __file__,
                'config': config,
                'model': config['model'],
                'model_args': config['model_args'],
                'model_sd': model_.state_dict(),
                'training': training,
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            
            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                        save_path, 'epoch-{}.pth'.format(epoch)))
            
            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))
        
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--res_type', default=None)
    parser.add_argument('--teacher_res_type', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    init_env()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    
    if args.res_type:
        ori_res_type = config['model_args']['encoder']
        config['model_args']['encoder'] = args.res_type
        print(f'Change encoder res_type from {ori_res_type} to {args.res_type}')
    
    if args.teacher_res_type:
        ori_teach_res_type = config['teacher_model_args']['encoder']
        config['teacher_model_args']['encoder'] = args.teacher_res_type
        config['teacher_load'] = config['teacher_load'].replace(ori_teach_res_type, args.teacher_res_type)
        print(f'Change teacher res_type from {ori_teach_res_type} to {args.teacher_res_type}')
    
    utils.set_gpu(args.gpu)
    main(config)
