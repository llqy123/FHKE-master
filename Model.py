import math
import torch
import torch.nn as nn
from geoopt import ManifoldParameter
from manifolds import Lorentz

class FHKE(torch.nn.Module):
    def __init__(self, d, dim, max_scale, max_norm, margin):
        super(FHKE, self).__init__()
        self.manifold = Lorentz(max_norm=max_norm)

        self.emb_entity = ManifoldParameter(self.manifold.random_normal((len(d.entities), dim), std=1./math.sqrt(dim)), manifold=self.manifold)
        self.relation_bias = nn.Parameter(torch.zeros((len(d.relations), dim)))
        self.diag = nn.Parameter(torch.empty(len(d.relations), dim))
        nn.init.kaiming_uniform_(self.diag)
        self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def givens_rotations(self, r, x, bias=None):
        givens = r.view((r.shape[0], -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], -1, 2))
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        x = x_rot.view((r.shape[0], -1))
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale + 1.1
        if bias is not None:
            x = x + bias
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
        x = torch.cat([time, x_narrow], dim=-1)
        return x

    def forward(self, u_idx, r_idx, v_idx):
        u_idx = u_idx.type(torch.long)
        r_idx = r_idx.type(torch.long)
        v_idx = v_idx.type(torch.long)
        h = self.emb_entity[u_idx]
        t = self.emb_entity[v_idx]
        r_bias = self.relation_bias[r_idx]
        r_diag = self.diag[r_idx]
        h = self.givens_rotations(r_diag, h, r_bias)
        neg_dist = (self.margin + 2 * self.manifold.cinner(h.unsqueeze(1), t).squeeze(1))

        return neg_dist + self.bias_head[u_idx].unsqueeze(-1) + self.bias_tail[v_idx]