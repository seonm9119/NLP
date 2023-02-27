import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import random
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

"TODO : Build LSTMmodel"
class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        lstm_cell_list = nn.ModuleList()
        lstm_cell_list.append(nn.LSTMCell(input_size, hidden_size))
        for l in range(1, num_layers):
            lstm_cell_list.append(nn.LSTMCell(hidden_size, hidden_size))

        self.lstm_cell_list = lstm_cell_list

    def forward(self, input, state=None):

        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)

            hx = Variable(torch.zeros(self.num_layers, max_batch_size, self.hidden_size).cuda())
            cx = Variable(torch.zeros(self.num_layers, max_batch_size, self.hidden_size).cuda())

            output = []

            first = 0
            for i in range(20):
                last = batch_sizes[i] + first
                for layer in range(self.num_layers):
                    hidden = hx[layer][:batch_sizes[i]].clone()
                    cell = cx[layer][:batch_sizes[i]].clone()
                    if layer == 0:
                        h, c = self.lstm_cell_list[layer](input[first:last], (hidden, cell))
                    else:
                        hidden_l = hx[layer-1][:batch_sizes[i]].clone()
                        h, c = self.lstm_cell_list[layer](hidden_l, (hidden, cell))
                    hx[layer][:batch_sizes[i]] = h
                    cx[layer][:batch_sizes[i]] = c
                first = last

                output = torch.concat((output,h), dim=0) if i else h


            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            hx = hx.index_select(1, unsorted_indices)
            cx = cx.index_select(1, unsorted_indices)
            return output_packed, (hx, cx)

        else:
            hx, cx = state
            output = []
            for i in range(input.size()[0]):
                for layer in range(self.num_layers):
                    hidden = hx[layer].clone()
                    cell = cx[layer].clone()
                    if layer == 0:
                        h, c = self.lstm_cell_list[layer](input[i], (hidden, cell))
                    else:
                        hidden_l = hx[layer-1].clone()
                        h, c = self.lstm_cell_list[layer](hidden_l, (hidden, cell))
                    hx[layer] = h
                    cx[layer] = c

                output.append(h)
            output = torch.stack(output, dim=0)
            return output, (hx, cx)

"TODO : Build Encoder-Decoder model"
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, auto=True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.auto = auto

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, mode='train'):

        batch_size = src.shape[0]
        trg_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_output, hidden, cell = self.encoder(src)

        eos = [2] * batch_size
        input = torch.IntTensor(eos).to(self.device)


        for t in range(0, trg_len):

            output, hidden, att = self.decoder(input, hidden, cell, encoder_output)
            outputs[t] = output

            top1 = output.argmax(1)

            "TO DO : Teacher-forcing"
            teacher_force = random.random() < teacher_forcing_ratio

            if mode == 'train':
                "TO DO : Autoregressive vs. Non-Autoregressive "
                "self.auto : True [Autoregressive + TF]"
                "self.auto : False [Non-Autoregressive]"
                input= trg[t] if teacher_force else top1 if self.auto else trg[t]
            else:
                input = top1


        return outputs

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = LSTMmodel(emb_dim, hid_dim, n_layers)


    def forward(self, src):

        inputs_length = torch.sum(torch.where(src > 0, True, False), dim=1)
        embedded = self.embedding(src)
        packed = pack(embedded, inputs_length.tolist(), batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.lstm(packed)
        output, outputs_length = unpack(output, batch_first=True)

        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = LSTMmodel(emb_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)


        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, output_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input, hidden, cell, encoder_outputs):

        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.classifier(output.squeeze(0))

        return prediction, hidden, cell


"TODO : Build Attention Decoder model"
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=20):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_dim = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = LSTMmodel(self.hidden_size*2, self.hidden_size, 4)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)


        hidden2 = hidden[0].unsqueeze(0)
        cat = torch.cat((embedded, hidden2), 2)
        attn_weights = F.softmax(self.attn(cat), dim=1)
        attn_weights = attn_weights.permute(1,0,2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.permute(1,0,2)


        output = torch.cat((embedded, attn_applied), 2)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights



