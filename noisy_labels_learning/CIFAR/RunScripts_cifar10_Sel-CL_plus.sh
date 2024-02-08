python3 train_Sel-CL_fine-tuning.py \
--noise_ratio 0.2 --noise_type "symmetric" --network "PR18" \
--experiment_name CIFAR10 --train_root ./dataset --out ./out

python3 train_Sel-CL_fine-tuning.py \
--noise_ratio 0.4 --noise_type "asymmetric" --network "PR18" \
--experiment_name CIFAR10 --train_root ./dataset --out ./out