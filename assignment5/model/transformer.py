import torch
import torch.nn as nn
from model.sub_layers import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.linear = nn.Linear(args.model_dim, args.num_token, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, src, tgt):

        enc = self.encoder(src)
        dec = self.decoder(tgt, src, enc)
        output = self.linear(dec)
        output = self.softmax(output)

        return output

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)



