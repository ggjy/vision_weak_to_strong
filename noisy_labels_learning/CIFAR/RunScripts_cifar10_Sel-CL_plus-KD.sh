python3 train_Sel-CL_fine-tuning.py \
--noise_ratio 0.2 --noise_type "symmetric" \
--dataset "CIFAR-10" --network "PR34" \
--teacher_network PR18 --kd_config kd_config/adapt_conf_kd_kw1_kt20_th05.yaml

python3 train_Sel-CL_fine-tuning.py \
--noise_ratio 0.4 --noise_type "asymmetric" \
--dataset "CIFAR-10" --network "PR34" \
--teacher_network PR18 --kd_config kd_config/adapt_conf_kd_kw1_kt10_th05.yaml