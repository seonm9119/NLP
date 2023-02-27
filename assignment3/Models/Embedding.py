import torch
import math


class CustomEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, _weight=None):
        super(CustomEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mat = torch.eye(num_embeddings)

        if _weight is None:
            self._weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, requires_grad=True))
            torch.nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            self._weight = torch.nn.Parameter(_weight)


    def forward(self, x):
        x = self.mat[x].cuda()
        emb = torch.matmul(x, self._weight)
        emb = emb.reshape((x.shape[0],x.shape[1],-1))
        return emb

    @classmethod
    def from_pretrained(cls, embeddings):

        embeddings = torch.FloatTensor(embeddings)
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
        embedding._weight.requires_grad = False
        return embedding


