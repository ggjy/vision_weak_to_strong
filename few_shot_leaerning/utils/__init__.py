import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from . import few_shot

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():
    
    def __init__(self):
        self.n = 0.0
        self.v = 0.0
    
    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n
    
    def item(self):
        return self.v


class Timer():
    
    def __init__(self):
        self.v = time.time()
    
    def s(self):
        self.v = time.time()
    
    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=False):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_') or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path, exist_ok=True)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()
    
    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)
    
    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)
    
    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
#                      start_warmup_value=0, warmup_steps=-1):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_steps > 0:
#         warmup_iters = warmup_steps
#     print("Set warmup steps = %d" % warmup_iters)
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = np.array(
#         [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

#     schedule = np.concatenate((warmup_schedule, schedule))

#     assert len(schedule) == epochs * niter_per_ep
#     return schedule


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    elif name == 'adamw':
        optimizer = AdamW(params, lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if milestones and name == 'adamw':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=milestones, eta_min=0.01 * lr)
    elif milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def kd_loss(logits_student, logits_teacher, temperature=1., reduction='batchmean'):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction=reduction)
    loss_kd *= temperature ** 2
    return loss_kd


def aug_conf_loss(logits_student, logits_teacher, kd_temperature, superalignment_threshold, epoch, total_epochs):
    if epoch < int(total_epochs * 0.25):
        alpha = epoch / (total_epochs * 0.25) * superalignment_threshold
    else:
        alpha = superalignment_threshold
    
    loss_kd = (1 - alpha) * kd_loss(logits_student, logits_teacher, kd_temperature)
    # Ablation with mixup
    loss_align = alpha * torch.nn.CrossEntropyLoss()(logits_student, torch.argmax(logits_student, dim=1))
    
    return loss_kd, loss_align


def adapt_conf_loss(logits_student, logits_teacher, kd_temperature, superalignment_threshold, epoch, total_epochs):
    ce_self = F.cross_entropy(logits_student, torch.argmax(logits_student, dim=1), reduction='none')
    ce_teacher = F.cross_entropy(logits_student, torch.argmax(logits_teacher, dim=1), reduction='none')
    ce_self = torch.exp(ce_self / superalignment_threshold)
    ce_teacher = torch.exp(ce_teacher / superalignment_threshold)
    alpha = (ce_self / (ce_self + ce_teacher)).detach()
    
    if epoch < int(total_epochs * 0.25):
        alpha = epoch / (total_epochs * 0.25) * alpha
    else:
        alpha = alpha
    
    loss_kd = (
            (1 - alpha) * kd_loss(logits_student, logits_teacher, kd_temperature, reduction='none').sum(1)
    ).mean(0)
    # Ablation with mixup
    loss_align = (
            alpha * torch.nn.CrossEntropyLoss(reduction='none')(logits_student, torch.argmax(logits_student, dim=1))
    ).mean(0)
    
    return loss_kd, loss_align
