import numpy as np

# Worked with Suhas Chundi on this
data = np.load('data/train.npz')

np.savez('data/smaller_train.npz', context_idxs=data['context_idxs'][:750],
         context_char_idxs=data['context_char_idxs'][:750],
         ques_idxs=data['ques_idxs'][:750],
         ques_char_idxs=data['ques_char_idxs'][:750],
         y1s=data['y1s'][:750],
         y2s=data['y2s'][:750],
         ids=data['ids'][:750])

dev = np.load('data/dev.npz')

np.savez('data/smaller_dev.npz', context_idxs=data['context_idxs'][:50],
         context_char_idxs=data['context_char_idxs'][:50],
         ques_idxs=data['ques_idxs'][:50],
         ques_char_idxs=data['ques_char_idxs'][:50],
         y1s=data['y1s'][:50],
         y2s=data['y2s'][:50],
         ids=data['ids'][:50])
