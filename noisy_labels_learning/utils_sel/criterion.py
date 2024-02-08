import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

def accuracy_v1(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    correct = [0] * len(top)
    _, labels_pred = torch.sort(preds, dim=1, descending=True)
    for idx, label_pred in labels_pred:
        result = (label_pred == labels[idx])
        j = 0
        for i in range(top[-1]):
            while i-1 > top[j]:
                j += 1
            if result[i] == 1:
                while j < len(top):
                    correct[j] += (100.0 / len(preds))
                # end the loop
                break
    return correct


def accuracy_v2(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result

def accuracy_v3(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k)

    return result
    
eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).

class KLDiv(nn.Module):
    # Calculate KL-Divergence
        
    def forward(self, predict, target):
       assert predict.ndimension()==2,'Input dimension must be 2'
       target = target.detach()

       # KL(T||I) = \sum T(logT-logI)
       predict += eps
       target += eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(1)
       return kld