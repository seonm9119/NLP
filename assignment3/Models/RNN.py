import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from Models.Embedding import CustomEmbedding

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=True)

        self.init_parameters()


    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        hy = (self.x2h(input) + self.h2h(hx))
        hy = torch.tanh(hy)

        return hy



class RNNmodel(nn.Module):
    def __init__(self, args, bidirectional=False):
        super(RNNmodel, self).__init__()

        self.args = args
        self.word_encoder = CustomEmbedding.from_pretrained(args.vectors).cuda()
        self.input_size = self.word_encoder.embedding_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.bidirectional = bidirectional

        def module_layers():
            rnn_cell_list = nn.ModuleList()
            rnn_cell_list.append(RNNCell(self.input_size, self.hidden_size))

            for l in range(1, self.num_layers):
                rnn_cell_list.append(RNNCell(self.hidden_size,self.hidden_size))
            return rnn_cell_list

        if self.bidirectional:
            self.forward_rnn = module_layers()
            self.backward_rnn = module_layers()
        else:
            self.rnn_cell_list = module_layers()

        self.fc = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.output_size)

    def forward(self, fx, bx=None, hx=None):

        if self.bidirectional:
            if bx is None:
                bx = fx
            return self.forward_both(fx, bx, hx)
        else:
            return self.forward_one(fx, hx)

    def forward_one(self, input, hx=None):

        input = self.word_encoder(input)

        if hx is None:
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
        else:
             h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

            outs.append(hidden_l)

        outs = outs[-1].squeeze()
        out = self.fc(outs)

        return out

    def forward_both(self, fx, bx, hx=None):

        fx = self.word_encoder(fx)
        bx = self.word_encoder(bx)

        if hx is None:
            f_h0 = Variable(torch.zeros(self.num_layers, fx.size(0), self.hidden_size).cuda())
            b_h0 = Variable(torch.zeros(self.num_layers, fx.size(0), self.hidden_size).cuda())
        else:
             f_h0 = hx
             b_h0 = hx

        f_outs = torch.zeros(fx.size(0), fx.size(1), self.hidden_size).cuda()
        b_outs = torch.zeros(fx.size(0), fx.size(1), self.hidden_size).cuda()

        hidden_f = list()
        hidden_b = list()
        for layer in range(self.num_layers):
            hidden_f.append(f_h0[layer, :, :])
            hidden_b.append(b_h0[layer, :, :])

        for t in range(fx.size(1)):
            for layer in range(self.num_layers):
                #forward
                if layer == 0:
                    f_h = self.forward_rnn[layer](fx[:, t, :], hidden_f[layer])
                else:
                    f_h = self.forward_rnn[layer](hidden_f[layer - 1],hidden_f[layer])
                hidden_f[layer] = f_h

                #backward
                if layer == 0:
                    b_h = self.backward_rnn[layer](bx[:, t, :], hidden_b[layer])
                else:
                    b_h = self.backward_rnn[layer](hidden_b[layer - 1],hidden_b[layer])
                hidden_b[layer] = b_h

            f_outs[:, t] = f_h
            b_outs[:, t] = b_h


        out = torch.concat([f_outs,b_outs],2)
        out = self.fc(out)

        return out