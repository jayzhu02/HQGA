import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer("position_ids",
                             torch.arange(max_position_embeddings).expand((1, -1)),
                             )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)

        return x + position_embeddings


class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True,
                 input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        :param dim_embed:
        :param dim_hidden:
        :param vocab_size:
        :param input_dropout_p:
        :param rnn_dropout_p:
        :param n_layers:
        :param bidirectional:
        :param rnn_cell:
        """
        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.max_qa_length = 37  # Same in sample_loader.py
        self.temporal_length = 12  # total number of category and signal in get_tce_and_tse() in sample_loader.py

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
            self.temporal_embedding = nn.Linear(self.temporal_length, input_dim)
            # nn.Embedding
            self.pe = LearnedPositionalEncoding(self.max_qa_length, input_dim, self.max_qa_length)

        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)

    def forward(self, qns, qns_lengths, temp_multihot=None, hidden=None):
        """
        2022.3.21@Jie Zhu
        Concatenate Category encoding and Signal encoding here.
        We should get category ans signal when loading the sample list. Go to sample_loader.py to add.

        :param qns:
        :param qns_lengths:
        :param temp_encoding: dim: batch* 1 * temporal_length
        :return:
        """

        # First do the positonal encoding
        if temp_multihot is not None:
            qns_encoding = self.pe(qns)
            temp_dim = self.temporal_embedding(temp_multihot)
            temp_dim = temp_dim.view(temp_dim.shape[0], 1, temp_dim.shape[1])
            #  Concatenate Temporal encoding
            qns = torch.cat((qns_encoding, temp_dim), 1)  # dim: batch * max(qa_length) +1 * 768

        qns_embed = self.embedding(qns)  # batch * max(qas_length)??? * dim_embed
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)  # ? * dim_embed
        packed_output, hidden = self.rnn(packed, hidden)  # ? * dim_hidden, 1 * batch * dim_hidden
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # batch * max(qns_lengths) * dim_hidden
        return output, hidden
