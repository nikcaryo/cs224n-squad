import numpy as np

# Worked with Suhas Chundi on this
data = np.load('data/train.npz')

np.savez('data/small_train.npz', context_idxs=data['context_idxs'][:1500],
         context_char_idxs=data['context_char_idxs'][:1500],
         ques_idxs=data['ques_idxs'][:1500],
         ques_char_idxs=data['ques_char_idxs'][:1500],
         y1s=data['y1s'][:1500],
         y2s=data['y2s'][:1500],
         ids=data['ids'][:1500])

dev = np.load('data/dev.npz')

np.savez('data/small_dev.npz', context_idxs=data['context_idxs'][:100],
         context_char_idxs=data['context_char_idxs'][:100],
         ques_idxs=data['ques_idxs'][:100],
         ques_char_idxs=data['ques_char_idxs'][:100],
         y1s=data['y1s'][:100],
         y2s=data['y2s'][:100],
         ids=data['ids'][:100])
