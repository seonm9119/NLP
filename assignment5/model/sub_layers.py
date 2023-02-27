import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#5. Positional Encoding
def position_encoding(max_len, model_dim):

    def angle_vec(position):
        return [position / np.power(10000, 2 * (i_hidn // 2) / model_dim)  for i_hidn in range(model_dim)]

    position_vec = np.array([angle_vec(i_seq) for i_seq in range(max_len)])
    position_vec[:, 0::2] = np.sin(position_vec[:, 0::2])
    position_vec[:, 1::2] = np.cos(position_vec[:, 1::2])

    return position_vec


#2.1. PAD mask
def pad_mask(Q, K, i_pad):
    batch_size, Q_s = Q.size()
    batch_size, K_s = K.size()
    pad_attn_mask = K.data.eq(i_pad).unsqueeze(1).expand(batch_size, Q_s, K_s)
    return pad_attn_mask


#2.2 Sub-Sequence mask
def subsequent_mask(tgt):
    subsequent_mask = torch.ones_like(tgt).unsqueeze(-1).expand(tgt.size(0), tgt.size(1), tgt.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)
    return subsequent_mask


#2. Scale Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.d_prob)
        self.scale = 1 / (self.args.d_k ** 0.5)

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)

        return context


#3. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fc_q = nn.Linear(args.model_dim, args.n_head * args.d_k)
        self.fc_k = nn.Linear(args.model_dim, args.n_head * args.d_k)
        self.fc_v = nn.Linear(args.model_dim, args.n_head * args.d_v)
        self.scaled_dot_attn = ScaledDotProductAttention(args)
        self.linear = nn.Linear(args.model_dim, args.model_dim)
        self.dropout = nn.Dropout(args.d_prob)

    def forward(self, Q, K, V, attn_mask):

        # 1. Set Query, Key, Value
        batch_size = Q.size(0)

        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q = Q.view(batch_size, -1, self.args.n_head, self.args.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.args.n_head, self.args.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.args.n_head, self.args.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_head, 1, 1)
        context = self.scaled_dot_attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.args.model_dim)

        output = self.linear(context)
        output = self.dropout(output)

        return output


#4. Feed Forward Layer
class FeedForwardNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_1 = nn.Linear(args.model_dim, args.hidden_size)
        self.fc_2 = nn.Linear(args.hidden_size, args.model_dim)
        self.dropout = nn.Dropout(args.d_prob)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.self_attn = MultiHeadAttention(args)
        self.layer_norm1 = nn.LayerNorm(args.model_dim, eps=args.layer_norm_epsilon)
        self.ffn = FeedForwardNet(args)
        self.layer_norm2 = nn.LayerNorm(args.model_dim, eps=args.layer_norm_epsilon)

    def forward(self, src, attn_mask):

        att_outputs = self.self_attn(src, src, src, attn_mask)
        att_outputs = self.layer_norm1(src + att_outputs)

        ffn_outputs = self.ffn(att_outputs)
        outputs = self.layer_norm2(ffn_outputs + att_outputs)
        return outputs



class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_emb = nn.Embedding(args.num_token, args.model_dim)

        # 5. Positional Encoding
        position_vec = torch.FloatTensor(position_encoding(args.max_len + 1, args.model_dim))
        self.pos_emb = nn.Embedding.from_pretrained(position_vec, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.num_enc_layers)])

    def forward(self, src):

        positions = torch.arange(src.size(1), device=src.device, dtype=src.dtype).expand(src.size(0), src.size(1)).contiguous() + 1
        pos_mask = src.eq(self.args.i_pad)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(src) + self.pos_emb(positions)
        attn_mask = pad_mask(src, src, self.args.i_pad)

        for layer in self.layers:
            outputs = layer(outputs, attn_mask)

        return outputs



class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.self_attn = MultiHeadAttention(args)
        self.layer_norm1 = nn.LayerNorm(args.model_dim, eps=args.layer_norm_epsilon)
        self.enc_attn = MultiHeadAttention(args)
        self.layer_norm2 = nn.LayerNorm(args.model_dim, eps=args.layer_norm_epsilon)
        self.ffn = FeedForwardNet(args)
        self.layer_norm3 = nn.LayerNorm(args.model_dim, eps=args.layer_norm_epsilon)

    def forward(self, tgt, enc, self_attn_mask, enc_attn_mask):

        self_att_outputs = self.self_attn(tgt, tgt, tgt, self_attn_mask)
        self_att_outputs = self.layer_norm1(tgt + self_att_outputs)

        enc_att_outputs = self.enc_attn(self_att_outputs, enc, enc, enc_attn_mask)
        enc_att_outputs = self.layer_norm2(self_att_outputs + enc_att_outputs)

        ffn_outputs = self.ffn(enc_att_outputs)
        outputs = self.layer_norm3(enc_att_outputs + ffn_outputs)

        return outputs



class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dec_emb = nn.Embedding(args.num_token, args.model_dim)
        position_vec = torch.FloatTensor(position_encoding(args.max_len + 1, args.model_dim))
        self.pos_emb = nn.Embedding.from_pretrained(position_vec, freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.num_dec_layers)])

    def forward(self, tgt, src, enc):

        positions = torch.arange(tgt.size(1), device=tgt.device, dtype=tgt.dtype).expand(
            tgt.size(0), tgt.size(1)).contiguous() + 1
        pos_mask = tgt.eq(self.args.i_pad)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.dec_emb(tgt) + self.pos_emb(positions)

        attn_pad_mask = pad_mask(tgt, tgt, self.args.i_pad)
        attn_decoder_mask = subsequent_mask(tgt)
        self_attn_mask = torch.gt((attn_pad_mask + attn_decoder_mask), 0)
        enc_attn_mask = pad_mask(tgt, src, self.args.i_pad)

        for layer in self.layers:
            outputs = layer(outputs, enc, self_attn_mask, enc_attn_mask)


        return outputs







