import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller


@register_distiller
class ReviewKD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(ReviewKD, self).__init__(student, teacher, criterion, args)
        
        self.shapes = [1, 8, 16, 32, 32]
        self.out_shapes = [1, 8, 16, 32, 32]
        in_channels = [16, 16, 32, 64, 64]
        out_channels = [16, 16, 32, 64, 64]
        self.max_mid_channel = 512

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)
            
        logits_student, feat_student = self.student(image, requires_feat=True)
        
        x = feat_student[:-1] + [feat_student[-1].unsqueeze(-1).unsqueeze(-1)]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        feat_teacher = feat_teacher[:-1] + [feat_teacher[-1].unsqueeze(-1).unsqueeze(-1)]
        # losses
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = (
            self.args.kd_loss_weight * hcl_loss(results, feat_teacher)
        )
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
    

def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all



class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x