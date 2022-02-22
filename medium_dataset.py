import numpy as np

# Worked with Suhas Chundi on this
data = np.load('data/train.npz')

np.savez('data/medium_train.npz', context_idxs=data['context_idxs'][:1000], 
                                    context_char_idxs=data['context_char_idxs'][:1000], 
                                    ques_idxs=data['ques_idxs'][:1000], 
                                    ques_char_idxs=data['ques_char_idxs'][:1000], 
                                    y1s=data['y1s'][:1000], 
                                    y2s=data['y2s'][:1000], 
                                    ids=data['ids'][:1000])
