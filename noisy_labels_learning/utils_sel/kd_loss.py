import torch
import torch.nn.functional as F


def init_kd_exp_name(config):
    kd_name = f"_{config['kd_loss_type']}-th{config['superalignment_threshold']}-kw{config['loss_weight']}-kt{config['temperature']}"
    return kd_name


def kd_loss(logits_student, logits_teacher, temperature=1., reduction='batchmean'):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction=reduction)
    loss_kd *= temperature ** 2
    return loss_kd


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


def get_kd_loss(config, logits, teacher_logits, epoch, max_epoch):
    if config['kd_loss_type'] == 'adapt_conf':
        loss_kd, loss_align = adapt_conf_loss(logits, teacher_logits,
                                              config['temperature'],
                                              config['superalignment_threshold'],
                                              epoch, max_epoch)
        loss = loss_kd + loss_align
    else:
        raise NotImplementedError(f'loss type {config["kd_loss_type"]} is not implemented')
    
    loss = config['loss_weight'] * loss
    
    return loss
