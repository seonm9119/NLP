import torch
import math
class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.voca_size = 2548
        self.embed_size = 1000
        self.mat = torch.eye(self.voca_size).cuda()

        self.weight = torch.nn.Parameter(torch.empty(self.voca_size, self.embed_size, requires_grad=True)).cuda()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))




    def forward(self, x):

        x = self.mat[x]
        self.embedding = torch.matmul(x, self.weight)

        return self.embedding.reshape((x.shape[0],-1))
