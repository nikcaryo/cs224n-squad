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
        print('--- Model used: BiDAF ---')
        print('--- Model is using the following layers --- \n')

        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        print('layers.Embedding')

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        print('layers.RNNEncoder')

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        print('layers.BiDAFAttention')

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        print('layers.RNNEncoder')

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)
        print('layers.BiDAFOutput \n')
        print('--- Time to train/test! --- \n')

    # match this forward to charbidaf to make train/eval easier
    def forward(self, cw_idxs, qw_idxs, cc_idxs=[], qc_idxs=[]):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # Lookup the word level embeddings + refine them with the highway network
        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        # Encode both the context and question with RNN
        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(att, c_len)

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

    def __init__(self, word_vectors, hidden_size, char_vectors, char_hidden_size, use_char, attention, drop_prob=0.2, output='rnet', attention_size=100):
        super(CharBiDAF, self).__init__()
        print('--- Model used: CharBiDAF ---')
        print('--- Model is using the following layers --- \n')

        if use_char:
            self.emb = layers.EmbeddingRNET(word_vectors=word_vectors,
                                            char_vectors=char_vectors,
                                            char_hidden_size=char_hidden_size,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob,
                                            num_layers=2)
            print('layers.EmbeddingRNET')
        else:
            self.emb = layers.Embedding(word_vectors=word_vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
            print('layers.Embedding')

        # get correct input size based on use of char embettings or not
        rnn_encoder_input_size = hidden_size
        if use_char:
            rnn_encoder_input_size = word_vectors.size(
                1) + 2 * char_hidden_size

        self.enc = layers.RNNEncoder(input_size=rnn_encoder_input_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        print('layers.RNNEncoder')

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        print('layers.BiDAFAttention')

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob,
                                     lstm=False)
        print('layers.RNNEncoder')

        self.attention_type = attention
        if self.attention_type == 'rnet':
            self.self_match = layers.SelfMatch2(
                hidden_size=hidden_size * 8, drop_prob=drop_prob)
            print('layers.SelfMatch2')

        self.output_type = output
        if self.output_type == 'bidaf':
            self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                          drop_prob=drop_prob)
            print('layers.BiDAFOutput \n')
        elif self.output_type == 'rnet':
            self.out = layers.OutputRNET(hidden_size=hidden_size,
                                         drop_prob=drop_prob,
                                         attention_size=attention_size,
                                         )
            print('layers.OutputRNET \n')
        print('--- Time to train/test! --- \n')

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)   # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)   # (batch_size, q_len, hidden_size)

        # Encode both the context and question with RNN
        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        # Equiv to gated attention
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        # print(att)
        # Self matching attention
        # (batch_size, c_len, 8 * hidden_size)
        if self.attention_type == 'rnet':
            self_match = self.self_match(att, att, c_mask, c_mask)
            att = self_match
 
        mod = self.mod(att, c_len)
        
        # self_match = torch.relu(self_match)
        # print(self_match.size())
        # print(self_match)
        # RNN as in RNET, but without the input dependent on output
        # (batch_size, c_len, 2 * hidden_size)
        
        # print(mod.size())

        if self.output_type == 'bidaf':
            # 2 tensors, each (batch_size, c_len)
            out = self.out(att, mod, c_mask)
        elif self.output_type == 'rnet':
            out = self.out(q_enc, mod)
        # print(out)

        return out
