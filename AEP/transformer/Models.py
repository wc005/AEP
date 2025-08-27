import math
import pickle
from transformer import Encoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, num_cascade_id, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.sig = nn.Sequential(
            nn.Sigmoid()
        )


        self.node_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        self.cascade_emb = nn.Embedding(num_cascade_id + 1, d_model, padding_idx=Constants.PAD)
        self.vaecoder = Encoder.VAE(num_types, d_model, d_model, d_model)
        self.encoder = Encoder.Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )
        self.base_encoder = Encoder.BasedEncoder(d_model)

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.nums_predictor = Predictor(d_model, 1)
        self.history = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.current = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.base = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.nums_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Softplus(),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def forward(self, event_type, event_time, time_factor, cascade_id):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = Encoder.get_non_pad_mask(event_type)
        node_emd = self.node_emb(event_type)
        cascade_emd = self.cascade_emb(cascade_id)
        vaecoder_output = self.vaecoder(node_emd)
        aftertime = vaecoder_output * time_factor.unsqueeze(2)
        # 历史强度编码
        enc_output = self.encoder(event_type, aftertime, event_time, non_pad_mask)
        history_output = self.rnn(enc_output, non_pad_mask)

        # 当前时刻编码
        current_factor = torch.sigmoid(event_time[:, -1])
        currentnode_en = aftertime[:, -1, :]
        current_encoding = currentnode_en * current_factor.unsqueeze(1)

        # 基本强度编码
        base_output = self.base_encoder(cascade_emd, vaecoder_output)
        base_emd = torch.prod(base_output, dim=1)
        ca_emd = self.history(history_output[:, -1, :]) + self.base(base_emd) + self.current(current_encoding)


        time_prediction = self.time_predictor(history_output, non_pad_mask)

        num_prediction = self.nums_predictor(ca_emd)

        return enc_output, (time_prediction, num_prediction)