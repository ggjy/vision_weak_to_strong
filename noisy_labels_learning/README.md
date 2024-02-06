
### **We follow [Sel-CL](https://github.com/ShikunLi/Sel-CL)'s code and add the knowledge distillation loss.**

## Result
Learning with noisy labels on CIFAR-10/CIFAR-100.

| teacher | student | Dataset   | noise type | top1  | top5   |
|---------|---------|-----------|------------|-------|--------|
| PR18    | PR34    | CIFAR-10  | asymmetric | 93.69 | 99.84  |
| PR18    | PR34    | CIFAR-10  | symmetric  | 96.13 | 99.87  |
| PR18    | PR34    | CIFAR-100 | asymmetric | 75.61 | 93.78  |
| PR18    | PR34    | CIFAR-100 | symmetric  | 78.64 | 94.03  |


## Requirements:
* Python 3.8.10
* Pytorch 1.8.0 (torchvision 0.9.0)
* Numpy 1.19.5
* scikit-learn 1.0.1
* apex 0.1


## Running the code on CIFAR-10/100:

- pre-train the backbone
  ```shell
  bash Sel-CL/CIFAR/RunScripts_cifar10_Sel-CL.sh
  bash Sel-CL/CIFAR/RunScripts_cifar100_Sel-CL.sh
  ```

- fine-tune the backbone
  ```shell
  bash Sel-CL/CIFAR/RunScripts_cifar10_Sel-CL_plus.sh
  bash Sel-CL/CIFAR/RunScripts_cifar100_Sel-CL_plus.sh
  ```

- weak-to-strong by knowledge distillation
  ```shell
  bash Sel-CL/CIFAR/RunScripts_cifar10_Sel-CL_plus-KD.sh
  bash Sel-CL/CIFAR/RunScripts_cifar100_Sel-CL_plus-KD.sh
  ```

To run the code use the provided scripts in CIFAR folders. Datasets are downloaded automatically when setting "--download True". The dataset has to be placed in dataset folder (should be done automatically). During training, the results can be obtained in the log file in out folder.

## Pretrain weight and log
You can find the knowledge distillation stage weights and logs from our released files.
