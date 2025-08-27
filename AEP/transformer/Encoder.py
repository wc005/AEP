import math
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer

class BasedEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.x_t = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
        ).cuda()
        self.n_t = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
        ).cuda()
        self.sig = nn.Sequential(
            nn.Sigmoid()
        ).cuda()
        self.para = torch.nn.Parameter(torch.Tensor(hidden_size).cuda())

    def forward(self, user, cascade):
        # 计算点过程历史隐变量分布

        aa = self.x_t(user) + self.n_t(cascade) + self.para
        gate = self.sig(aa)
        rrr = self.x_t(user) * gate
        return rrr

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])


    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_emb, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        for enc_layer in self.layer_stack:
            event_emb += tem_enc
            enc_output, _ = enc_layer(
                event_emb,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class VAE(nn.Module):
    def __init__(self, num_types, d_model, hidden_size, z_size):
        super().__init__()

        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        ).cuda()

        self.prior_encoder_mu = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus(),
        ).cuda()
        self.prior_encoder_std = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, z_size),
            nn.Softplus(),
        ).cuda()

        self.emd = nn.Sequential(
            nn.Linear(z_size + z_size, z_size),
            nn.Softplus(),
        ).cuda()


    def forward(self, x):
        # 计算点过程历史隐变量分布
        x = self.linear(x)
        prior_mu = self.prior_encoder_mu(x)
        prior_logvar = self.prior_encoder_std(x)
        Z = torch.cat((prior_mu, prior_logvar), dim=2)
        emd = self.emd(Z)
        return emd

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

