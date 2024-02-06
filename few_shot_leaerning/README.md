### **We follow [few-shot-meta-baseline](https://github.com/yinboc/few-shot-meta-baseline)'s code and add the knowledge distillation loss.**


## Result
Few shot learning on miniImageNet.

- Classifier stage

  | Teacher  | Student  | 1-shot | 5-shot  |
  |----------|----------|--------|---------|
  | ResNet12 | ResNet36 | 0.615  | 0.7952  |
  | ResNet18 | ResNet36 | 0.6229 | 0.7996  |

- meta-learning stage

  | teacher               | student  | val     |
  |-----------------------|----------|---------|
  | ResNet12 (classifier) | ResNet36 | 0.6538  |
  | ResNet18 (classifier) | ResNet36 | 0.6574  |
  | ResNet12 (meta)       | ResNet36 | 0.6608  |
  | ResNet18 (meta)       | ResNet36 | 0.6595  |


## Running the code

### Preliminaries

**Environment**

- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

**Datasets**

- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy
  of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))

Download the datasets and link the folders into `materials/` with names `mini-imagenet`.

**Environment variable**

You can setting the dataset path and output path in few-shot-meta-baseline/init_env.py

When running python programs, use `--gpu` to specify the GPUs for running the code (e.g. `--gpu 0,1`).
For Classifier-Baseline, we train with 4 GPUs on miniImageNet and tieredImageNet and with 8 GPUs on ImageNet-800.
Meta-Baseline uses half of the GPUs correspondingly.

In following we take miniImageNet as an example. For other datasets, replace `mini` with `tiered` or `im800`.
By default it is 1-shot, modify `shot` in config file for other shots. Models are saved in `save/`.

### 1. Training Classifier-Baseline

- Training
    ```
    python train_classifier.py --config config/classifier/train_classifier_mini.yaml --res_type resnet12_bottle --gpu 0,1,2,3
    python train_classifier.py --config config/classifier/train_classifier_mini.yaml --res_type resnet18_bottle --gpu 0,1,2,3
    python train_classifier.py --config config/classifier/train_classifier_mini.yaml --res_type resnet36_bottle --gpu 0,1,2,3
    ```
- Knowledge Distillation
  ```
  python train_classifier.py --config config/classifier/train_classifier_mini_kd_adapt_conf.yaml --res_type resnet36_bottle --teacher_res_type resnet12_bottle --gpu 0,1,2,3
  python train_classifier.py --config config/classifier/train_classifier_mini_kd_adapt_conf.yaml --res_type resnet36_bottle --teacher_res_type resnet18_bottle --gpu 0,1,2,3
  ```
  

### 2. Training Meta-Baseline

- Training
  ```
  python train_meta.py --config config/meta/train_meta_mini.yaml --res_type resnet12 --gpu 0,1
  python train_meta.py --config config/meta/train_meta_mini.yaml --res_type resnet18 --gpu 0,1
  python train_meta.py --config config/meta/train_meta_mini.yaml --res_type resnet36 --gpu 0,1
  ```
- Knowledge Distillation (classifier teacher)
  ```
  python train_meta.py --config config/meta/train_meta_mini_kd_adapt_conf.yaml --res_type resnet36_bottle --teacher_res_type resnet12_bottle --gpu 0,1
  python train_meta.py --config config/meta/train_meta_mini_kd_adapt_conf.yaml --res_type resnet36_bottle --teacher_res_type resnet18_bottle --gpu 0,1
  ```

- Knowledge Distillation (meta teacher)
  ```
  python train_meta.py --config config/meta/train_meta_mini_kd_adapt_conf.yaml --res_type resnet36_bottle --teacher_res_type resnet12_bottle  --teacher_meta_model --gpu 0,1
  python train_meta.py --config config/meta/train_meta_mini_kd_adapt_conf.yaml --res_type resnet36_bottle --teacher_res_type resnet18_bottle  --teacher_meta_model --gpu 0,1
  ```

### 3. Test

To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of
Classifier-Baseline, or setting `load` to the saving file of Meta-Baseline.

E.g., `load: ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth`

Then run

```
python test_few_shot.py --shot 1
```

## Advanced instructions

### Configs

A dataset/model is constructed by its name and args in a config file.

For a model, to load it from a specific saving file, change `load_encoder` or `load` to the corresponding path.
`load_encoder` refers to only loading its `.encoder` part.

In configs for `train_classifier.py`, `fs_dataset` refers to the dataset for evaluating few-shot performance.

In configs for `train_meta.py`, both `tval_dataset` and `val_dataset` are validation datasets, while `max-va.pth` refers
to the one with best performance in `val_dataset`.

### Single-class AUC

To evaluate the single-class AUC, add `--sauc` when running `test_few_shot.py`.

### Pretrain weight and log
You can find the knowledge distillation stage weights and logs from our released files.
