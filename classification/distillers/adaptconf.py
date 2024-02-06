import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import kd_loss


@register_distiller
class AugConf(BaseDistiller):
    requires_feat = False
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(AugConf, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)
        
        if self.args.epoch < int(self.args.epochs * self.args.superalignment_rate):
            alpha = self.args.epoch / (self.args.epochs * self.args.superalignment_rate) * self.args.superalignment_threshold
        else:
            alpha = self.args.superalignment_threshold

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * (1-alpha) * kd_loss(logits_student, logits_teacher, self.args.kd_temperature, reduction='batchmean')
        loss_align = self.args.kd_loss_weight * alpha * torch.nn.CrossEntropyLoss()(logits_student, torch.argmax(logits_student, dim=1))

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_align": loss_align,
        }
        return logits_student, losses_dict

@register_distiller
class AdaptConf(BaseDistiller):
    requires_feat = False
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(AdaptConf, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)
        
        ce_self = F.cross_entropy(logits_student, torch.argmax(logits_student, dim=1), reduction='none')
        ce_teacher = F.cross_entropy(logits_student, torch.argmax(logits_teacher, dim=1), reduction='none')
        ce_self = torch.exp(ce_self/self.args.superalignment_threshold)
        ce_teacher = torch.exp(ce_teacher/self.args.superalignment_threshold)
        alpha = (ce_self / (ce_self + ce_teacher)).detach()
        
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * (
            (1 - alpha) * kd_loss(logits_student, logits_teacher, self.args.kd_temperature, reduction='none').sum(1)
        ).mean(0)
        loss_align = self.args.kd_loss_weight * (
            alpha * torch.nn.CrossEntropyLoss(reduction='none')(logits_student, torch.argmax(logits_student, dim=1))
        ).mean(0)

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_align": loss_align,
        }
        return logits_student, losses_dict
