import os
import threading
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from timm.data import ImageDataset
from timm.utils.model import get_state_dict
from torchvision.datasets import CIFAR100


class ImageNetInstanceSample(ImageDataset):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR100InstanceSample(CIFAR100, ImageNetInstanceSample):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR100.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 100
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class MASaver:
    def __init__(self, local_path, enable, tmp_dir='tmp_checkpoint'):
        self.enabled = enable
        if self.enabled:
            self.local_path = os.path.join(local_path, tmp_dir)
            self.checkpoint = dict()

    def update(self, saver, epoch, metric=None):
        if not self.enabled or saver is None:
            return
        save_state = {
            'epoch': epoch,
            'arch': type(saver.model).__name__.lower(),
            'state_dict': get_state_dict(saver.model, saver.unwrap_fn),
            'optimizer': saver.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if saver.args is not None:
            save_state['arch'] = saver.args.model
            save_state['args'] = saver.args
        if saver.amp_scaler is not None:
            save_state[saver.amp_scaler.state_dict_key] = saver.amp_scaler.state_dict()
        if saver.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(saver.model_ema, saver.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        self.checkpoint[saver.checkpoint_dir.split('/')[-1]] = save_state

    def save(self, tmp_name='last_checkpoint.pth'):
        if not self.enabled:
            return
        try:
            torch.save(self.checkpoint, os.path.join(self.local_path, tmp_name))
        except FileNotFoundError:
            os.makedirs(self.local_path, exist_ok=True)
            torch.save(self.checkpoint, os.path.join(self.local_path, tmp_name))
        sync_thread.start()


class TimePredictor:
    def __init__(self, steps, most_recent=30, drop_first=True):
        self.init_time = time.time()
        self.steps = steps
        self.most_recent = most_recent
        self.drop_first = drop_first  # drop iter 0

        self.time_list = []
        self.temp_time = self.init_time

    def update(self):
        time_interval = time.time() - self.temp_time
        self.time_list.append(time_interval)

        if self.drop_first and len(self.time_list) > 1:
            self.time_list = self.time_list[1:]
            self.drop_first = False

        self.time_list = self.time_list[-self.most_recent:]
        self.temp_time = time.time()

    def get_pred_text(self):
        single_step_time = np.mean(self.time_list)
        end_timestamp = self.init_time + single_step_time * self.steps
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')
