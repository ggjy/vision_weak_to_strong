python3 train_Sel-CL_fine-tuning.py \
--dataset 'CIFAR-100' --num_classes 100 \
--noise_ratio 0.2 --noise_type "symmetric" \
--network "PR18"

python3 train_Sel-CL_fine-tuning.py \
--dataset 'CIFAR-100' --num_classes 100 \
--noise_ratio 0.4 --noise_type "asymmetric" \
--network "PR18"