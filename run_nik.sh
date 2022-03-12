#python3 train.py --name 'BSB_LR_0.5_HIDDEN_100_DROP_0.2' --model 'charbidaf' --num_workers 0 --lr 0.5 --use_char false --output bidaf --attention rnet
#python3 train.py --name 'BBR_LR_0.5_HIDDEN_100_DROP_0.2' --model 'charbidaf' --num_workers 0 --lr 0.5 --use_char false --output rnet --attention bidaf

python3 train.py --name 'CSR_LR_0.5_HIDDEN_130_DROP_0.4' --model 'charbidaf' --lr 0.5 --use_char true --output rnet --attention rnet --hidden_size 130 --drop_prob 0.4
