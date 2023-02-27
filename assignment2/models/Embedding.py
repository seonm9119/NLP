import torch
import math

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # STEP 3. Word Representation using Character Embedding
        self.char_size = 54
        self.embed_size = 100


        self.weight = torch.nn.Parameter(torch.empty(self.char_size, self.embed_size, requires_grad=True)).cuda()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))



    def forward(self, x):

        self.char_embedding = torch.matmul(x, self.weight)
        out = self.char_embedding.flatten(2)

        return out