"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

from unicodedata import bidirectional

from numpy import s_
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from util import masked_softmax, get_available_devices

device, _ = get_available_devices()

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb

class EmbeddingRNET(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activation for char-level embedding of word.
        drop_prob (float): Probability of zero-ing out activations
        num_layers (int): Number of layers for char-level RNN encoder.
    """
    def __init__(self, word_vectors, char_vectors, drop_prob, num_layers, char_hidden_size, hidden_size):
        super(EmbeddingRNET, self).__init__()
        self.drop_prob = drop_prob
        self.embed_word = nn.Embedding.from_pretrained(word_vectors)
        self.embed_char = nn.Embedding.from_pretrained(char_vectors)
        self.char_vectors = char_vectors
        self.num_layers = num_layers
        self.char_hidden_size = char_hidden_size

        emb_size = word_vectors.size(1) + 2 * char_hidden_size

        self.proj = nn.Linear(emb_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

        self.char_encoder = nn.GRU(
            input_size = char_vectors.size(1),
            hidden_size = self.char_hidden_size,
            num_layers = self.num_layers,
            dropout = self.drop_prob,
            bidirectional = True,
            batch_first = True,
        )

    def forward(self, w_idxs, c_idxs):

        emb_word = self.embed_word(w_idxs)   # (batch_size, seq_len, embed_size)
        emb_chars = self.embed_char(c_idxs)  # (batch_size, seq_char_len, max_chars, char_embed_size)

        emb_word = F.dropout(emb_word, self.drop_prob)
        emb_chars = F.dropout(emb_chars, self.drop_prob)

        # reshape so that a 'batch' is a single word, 
        batch_size, seq_len, max_len, char_embed_size = emb_chars.size()
        emb_chars = emb_chars.view(batch_size * seq_len, max_len, char_embed_size)

        # for each word, feed each char into the rnn
        _, hn = self.char_encoder(emb_chars)

        # reshape so that we match the first two dims of emb_words
        # meaning, for each batch, for each seq, each learned char-word embedding
        # where the char-word embedding is the last hidden state of the last rnn layer
        hn_last_forward = hn[self.num_layers - 1, :, :] # (batch_size, char_hidden_size)
        hn_last_forward = hn_last_forward.view(batch_size, seq_len, self.char_hidden_size)

        hn_last_backward = hn[self.num_layers, :, :] # (batch_size, char_hidden_size)
        hn_last_backward = hn_last_backward.view(batch_size, seq_len, self.char_hidden_size)
        
        # concat to get (batch_size, seq_len, word_embed_size + 2 * char_hidden_size)
        emb = torch.cat((emb_word, hn_last_forward, hn_last_backward), 2)

        return emb



class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class GatedAttention(nn.Module):
    def __init__(self, hidden_size, output_size, attention_size, drop_prob):
        super(GatedAttention, self).__init__()
        self.GRU = nn.GRU(hidden_size * 4, hidden_size * 2, batch_first = True, dropout=drop_prob, bidirectional=False)

        self.Wuq = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.Wup = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.Wvp = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

        self.Wg = nn.Linear(hidden_size * 4, hidden_size * 4, bias=False)
        

        self.v = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

        self.drop_prob = drop_prob
        self.output_size = output_size
        
    def forward(self, passage, question, mask=None):
        batch_size, passage_len, hidden_size = passage.size()
        # print('passage size: ', passage.size())
        _, question_len, _ = question.size()
        # print('question size: ', question.size())

        v = None
        vtp = Variable(torch.zeros(1, batch_size, hidden_size))
        vtp = vtp.to(device)


        for i in range(passage_len):
            p_word = passage[:,i,:]
            # print('p_word size: ', p_word.size())

            a = self.Wuq(question)
            # print('wuq size: ', a.size())
            b = self.Wup(p_word)
            # print('wup size: ', b.size())
            c = self.Wvp(vtp)
            # print('wvp size: ', c.size())
            temp = a.permute(1,0,2) + b
            # print('temp size', temp.size())
            s = self.v(torch.tanh(temp + c))
            # print('s size: ', s.size())
            s = s.permute(1, 0, 2)
            a_t = F.softmax(s, dim=0)
            # print('a_t size:',  a_t.size())
            c_t = torch.sum(a_t * question, dim=1)
            # print('c_t size: ', c_t.size())

            # gate
            passage_attn = torch.cat([p_word, c_t], dim = 1)
            # print('passage_attn size: ', passage_attn.size())
            gt = torch.sigmoid(self.Wg(passage_attn))
            # print('gt size: ', gt.size())
            rnn_input = gt * passage_attn

            # unsqueeze?
            rnn_input = rnn_input.unsqueeze(1)
            # print('rnn_input size: ', rnn_input.size())


            output, vtp = self.GRU(rnn_input, vtp)
            
            # print('vtp size part 2: ', vtp.size())

            if v is None:
                v = vtp
            else:
                v = torch.cat((v, vtp), dim=0)
        v = v.permute(1, 0, 2)
        # print('final v size:', v.size())

        return v


class SelfMatching(nn.Module):
    def __init__(self, hidden_size, output_size, attention_size, drop_prob):
        super(SelfMatching, self).__init__()
        self.GRU = nn.GRU(hidden_size * 4, hidden_size * 2, batch_first = True, dropout=drop_prob, bidirectional=True)

        self.Wvp1 = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.Wvp2 = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.v = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

        self.Wg = nn.Linear(hidden_size * 4, hidden_size * 4, bias=False)
        
        self.drop_prob = drop_prob
        self.output_size = output_size
        
    def forward(self, passage, mask=None):
        batch_size, passage_len, hidden_size = passage.size()
        # print('passage size: ', passage.size())

        v = None
        last_hidden = Variable(torch.zeros(2, batch_size, hidden_size))
        last_hidden = last_hidden.to(device)

        for i in range(passage_len):
            p_word = passage[:,i,:]
            # print('p_word size: ', p_word.size())

            #last_hidden = torch.cat((last_hidden[0,:,:], last_hidden[1, :, :]), dim=1)

            a = self.Wvp1(passage)
            # print('Wvp1 size: ', a.size())
            b = self.Wvp2(p_word)
            # print('Wvp2 size: ', b.size())

            temp = a.permute(1,0,2) + b
            # print('temp size', temp.size())
            s = self.v(torch.tanh(temp))
            # print('s size: ', s.size())
            s = s.permute(1, 0, 2)
            a_t = F.softmax(s, dim=0)
            # print('a_t size:',  a_t.size())
            c_t = torch.sum(a_t * passage, dim=1)
            # print('c_t size: ', c_t.size())

            # gate
            passage_attn = torch.cat([p_word, c_t], dim = 1)
            # print('passage_attn size: ', passage_attn.size())
            gt = torch.sigmoid(self.Wg(passage_attn))
            # print('gt size: ', gt.size())
            rnn_input = gt * passage_attn

            # unsqueeze?
            rnn_input = rnn_input.unsqueeze(1)
            # print('rnn_input size: ', rnn_input.size())


            output, last_hidden = self.GRU(rnn_input, last_hidden)
            last_hidden_temp = torch.cat((last_hidden[0,:,:], last_hidden[1, :, :]), dim=1)
            last_hidden_temp = last_hidden_temp.unsqueeze(0)
            # print('vtp size part 2: ', last_hidden.size())

            if v is None:
                v = last_hidden_temp
            else:
                v = torch.cat((v, last_hidden_temp), dim=0)
        v = v.permute(1, 0, 2)
        # print('final v size:', v.size())
        return v





class OutputRNET(nn.Module):
    """Output layer used by RNET for Question Answering

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob, attention_size):
        super(OutputRNET, self).__init__()
        self.W_h_P = nn.Linear(4 * hidden_size, attention_size)
        self.W_h_a = nn.Linear(2 * hidden_size, attention_size)
        self.v_T = nn.Linear(attention_size, 1)
        self.gru = nn.GRUCell(4 * hidden_size, 2 * hidden_size)



        # input size = 2 * hidden_size
        self.W_u_Q = nn.Linear(2 * hidden_size, attention_size)
        self.v_T_r = nn.Linear(attention_size, 1)

    def forward(self, q_enc, x):

        # initial state: rQ = h_init
        # x = x.permute(1, 0, 2)
        # q_enc = q_enc.permute(1, 0, 2)

        s = torch.tanh(self.W_u_Q(q_enc))
        s = self.v_T_r(s)
        # print('s size', s.size())
        a = F.softmax(s, dim=1)
        # print('a size', a.size())
        # print('q_enc', q_enc.size())
        h_init = torch.sum(a * q_enc, dim=1)
        h_init = h_init.unsqueeze(1)
        # print('h init size', h_init.size())

        # print('x', x.size())
        s_1 = self.W_h_P(x)
        s_2 = self.W_h_a(h_init)

        # print('s_1', s_1.size())
        # print('s_2', s_2.size())

        s_start = torch.tanh(s_1+ s_2)
        # print('s_start', s_start.size())
        s_start = self.v_T(s_start)
        log_p_start = F.log_softmax(s_start, dim=1)
        p_start = F.softmax(s_start, dim=1)

        cell_input = torch.sum(p_start * x, dim=1)

        # print('cell input', cell_input.size())
        h_init = h_init.squeeze(1)
        h_last = self.gru(cell_input, h_init)
        h_last = h_last.unsqueeze(1)

        s_end = torch.tanh(self.W_h_P(x) + self.W_h_a(h_last))
        s_end = self.v_T(s_end)
        log_p_end = F.log_softmax(s_end, dim=1)

        # print('log p end', log_p_end.size())




        # (batch_size, c_len) * 2
        return log_p_start.squeeze(-1), log_p_end.squeeze(-1)