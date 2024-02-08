python3 train_Sel-CL.py --dataset "CIFAR-100" --num_classes 100 --queue_per_class 100 \
--noise_ratio 0.2 --noise_type "symmetric" --network "PR18" \
--alpha 0.75 --beta 0.35


python3 train_Sel-CL.py --dataset "CIFAR-100" --num_classes 100 --queue_per_class 100 \
--noise_ratio 0.4 --noise_type "asymmetric" --network "PR18" \
--alpha 0.25 --beta 0.0
