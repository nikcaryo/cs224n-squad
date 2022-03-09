python3 train.py --name 'BSB_LR_0.5_HIDDEN_100_DROP_0.2' --model 'charbidaf' --num_workers 0 --lr 0.5 --use_char false --output bidaf --attention rnet
python3 train.py --name 'BBR_LR_0.5_HIDDEN_100_DROP_0.2' --model 'charbidaf' --num_workers 0 --lr 0.5 --use_char false --output rnet --attention bidaf
