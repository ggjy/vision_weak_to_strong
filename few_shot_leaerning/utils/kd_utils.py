import os

import torch

import utils
from models import models


def _init_kd_svname(config):
    kd_loss_config = config['kd_loss_config']
    loss_type = kd_loss_config['loss_type']
    if loss_type in ['aug_conf', 'adapt_conf']:
        svname = f"_{loss_type}-th{kd_loss_config['threshold']}-kw{kd_loss_config['loss_weight']}-kt{kd_loss_config['temperature']}"
    else:
        svname = f"_{loss_type}-kw{loss_type['loss_weight']}-kt{loss_type['temperature']}"
    return svname


def init_classifier_kd_svname(config):
    svname = f"classifier_{config['train_dataset']}_{config['teacher_model_args']['encoder']}-kd-{config['model_args']['encoder']}"
    svname += _init_kd_svname(config)
    return svname


def init_meta_kd_svname(config):
    svname = f"meta_{config['train_dataset']}-{config['n_shot']}shot_{config['model']}_{config['teacher_model_args']['encoder']}-kd-{config['model_args']['encoder']}"
    svname += _init_kd_svname(config)
    return svname


def init_class_teacher_model(config):
    teacher_model = models.make(config['teacher_model'], **config['teacher_model_args'])
    teacher_model = teacher_model.eval()
    
    save_root = os.getenv('OUT_PATH', './save')
    teacher_load_path = os.path.join(save_root, config['teacher_load'])
    
    msg = teacher_model.load_state_dict(torch.load(teacher_load_path)['model_sd'])
    utils.log('Teacher load from: {}, {}'.format(teacher_load_path, msg))
    
    return teacher_model


def init_meta_teacher_model(config):
    teacher_model = models.make(config['teacher_model'], **config['teacher_model_args'])
    teacher_model = teacher_model.eval()
    
    save_root = os.getenv('OUT_PATH', './save')
    teacher_load_path = os.path.join(save_root, config['teacher_load'])
    
    encoder = models.load(torch.load(teacher_load_path)).encoder
    msg = teacher_model.encoder.load_state_dict(encoder.state_dict())
    
    utils.log('Teacher load from: {}, {}'.format(teacher_load_path, msg))
    
    return teacher_model


def kd_epoch_init(config, aves_keys):
    aves_keys.append('tlkd')
    if config.get('kd_loss_config') and config['kd_loss_config']['loss_type'] in ['aug_conf', 'adapt_conf']:
        aves_keys.append('tlalign')
    
    return aves_keys


def get_kd_loss(config, logits, teacher_logits, epoch, max_epoch, aves, loss):
    kd_config = config['kd_loss_config']
    
    if kd_config['loss_type'] == 'adapt_conf':
        loss_kd, loss_align = utils.adapt_conf_loss(logits, teacher_logits,
                                                    kd_config['temperature'],
                                                    kd_config['threshold'],
                                                    epoch, max_epoch)
        loss += kd_config['loss_weight'] * (loss_kd + loss_align)
        aves['tlkd'].add(loss_kd.item())
        aves['tlalign'].add(loss_align.item())
    elif kd_config['loss_type'] == 'kd':
        loss_kd = utils.kd_loss(logits, teacher_logits, temperature=config['temperature'])
        loss += config['loss_weight'] * loss_kd
        aves['tlkd'].add(loss_kd.item())
    
    return loss, aves
