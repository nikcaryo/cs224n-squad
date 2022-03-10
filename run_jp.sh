# python3 train.py --name 'CBB_LR_0.5_HIDDEN_100_DROP_0.5' --model 'charbidaf' --lr 0.5 --use_char true --output bidaf --attention bidaf --drop_prob 0.5
python3 train.py --name 'BSB_LR_0.5_HIDDEN_100_DROP_0.5' --model 'charbidaf' --lr 0.5 --use_char false --output bidaf --attention rnet --drop_prob 0.5
python3 train.py --name 'BBR_LR_0.5_HIDDEN_100_DROP_0.5' --model 'charbidaf' --lr 0.5 --use_char false --output rnet --attention bidaf --drop_prob 0.5
python3 train.py --name 'CSR_LR_0.5_HIDDEN_100_DROP_0.5' --model 'charbidaf' --lr 0.5 --use_char true --output rnet --attention rnet --drop_prob 0.5
