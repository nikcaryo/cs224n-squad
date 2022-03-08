"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.,):
        super(BiDAF, self).__init__()

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
       

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    # match this forward to charbidaf to make train/eval easier
    def forward(self, cw_idxs, qw_idxs, cc_idxs = [], qc_idxs = []):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # Lookup the word level embeddings + refine them with the highway network 
        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)


        # Encode both the context and question with RNN 
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)


        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out



class CharBiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, char_vectors, char_hidden_size, drop_prob=0. , output='rnet', attention_size=100):
        super(CharBiDAF, self).__init__()

        print('enc input size', word_vectors.size(1) + hidden_size)
        self.emb = layers.EmbeddingRNET(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    char_hidden_size=char_hidden_size,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    num_layers=2)
            
        self.enc = layers.RNNEncoder(input_size= word_vectors.size(1) + 2 * char_hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob,
                                     lstm=True)

        self.self_match = layers.SelfMatch2(hidden_size=hidden_size * 8, drop_prob=drop_prob)

        self.output = 'bidaf'
        if self.output == 'bidaf':
            self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        elif output == 'rnet':
            self.out = layers.OutputRNET(hidden_size=hidden_size,
                                        drop_prob=drop_prob,
                                        attention_size=attention_size,
                )



    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)   # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)   # (batch_size, q_len, hidden_size)

        # Encode both the context and question with RNN 
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # Equiv to gated attention
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        # print(att)
        # Self matching attention
        self_match = self.self_match(att, att, c_mask, c_mask) # (batch_size, c_len, 8 * hidden_size)
        # self_match = torch.relu(self_match)
        # print(self_match.size())
        # print(self_match)
        # RNN as in RNET, but without the input dependent on output
        mod = self.mod(self_match, c_len)        # (batch_size, c_len, 2 * hidden_size)
        # print(mod.size())
        
        if self.output == 'bidaf':
            out = self.out(self_match, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        elif self.output == 'rnet':
            out = self.out(q_enc, mod)
        # print(out)

        return out
