from typing import Tuple

import torch
import numpy as np

from models.TKBCModel import TKBCModel


class LowFERT(TKBCModel):
    def __init__(
        self,
        sizes: Tuple[int, int, int, int],
        rank: int,
        drop_rates: Tuple[float, float, float],
        no_time_emb=False,
        is_cuda: bool = True,
        k: int = 30,
    ) -> None:
        super(LowFERT, self).__init__()
        self.sizes = sizes
        self.no_time_emb = no_time_emb
        self.dim = rank

        self.E = torch.nn.Embedding(sizes[0], self.dim, padding_idx=0)
        self.R = torch.nn.Embedding(sizes[1], self.dim, padding_idx=0)
        self.R_noT = torch.nn.Embedding(sizes[1], self.dim, padding_idx=0)
        self.T = torch.nn.Embedding(sizes[3], self.dim, padding_idx=0)
        self.k, self.o = k, rank

        if is_cuda:
            self.U = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-0.01, 0.01, (self.dim, self.k * self.o)),
                    dtype=torch.float,
                    device="cuda",
                    requires_grad=True,
                )
            )
            self.V = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-0.01, 0.01, (self.dim, self.k * self.o)),
                    dtype=torch.float,
                    device="cuda",
                    requires_grad=True,
                )
            )
        else:
            self.U = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-0.01, 0.01, (self.dim, self.k * self.o)),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )
            self.V = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-0.01, 0.01, (self.dim, self.k * self.o)),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )

        self.input_dropout = torch.nn.Dropout(drop_rates[0])
        self.hidden_dropout1 = torch.nn.Dropout(drop_rates[1])
        self.hidden_dropout2 = torch.nn.Dropout(drop_rates[2])

        self.bn0 = torch.nn.BatchNorm1d(self.dim)
        self.bn1 = torch.nn.BatchNorm1d(self.dim)

        self.m = torch.nn.PReLU()

        torch.nn.init.kaiming_uniform_(self.E.weight.data)
        torch.nn.init.kaiming_uniform_(self.R.weight.data)
        torch.nn.init.kaiming_uniform_(self.T.weight.data)
        torch.nn.init.kaiming_uniform_(self.R_noT.weight.data)

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.R_noT(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])

        rel_t = rel * time + rel_no_time

        lhs = self.bn0(lhs)
        lhs = self.input_dropout(lhs)

        x = torch.mm(lhs, self.U) * torch.mm(rel_t, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.m(x)

        target = x * rhs

        return torch.sum(target, 1, keepdim=True)

    def forward(self, x):
        lhs = self.E(x[:, 0])
        rel = self.R(x[:, 1])
        rel_no_time = self.R_noT(x[:, 1])
        rhs = self.E(x[:, 2])
        time = self.T(x[:, 3])
        E = self.E.weight

        rel_t = rel * time + rel_no_time

        lhs = self.bn0(lhs)
        lhs = self.input_dropout(lhs)

        x = torch.mm(lhs, self.U) * torch.mm(rel_t, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.m(x)

        x = x @ E.t()
        pred = x

        return (pred, (
            torch.norm(lhs, p=4),
            torch.norm((rel * time), p=4),
            torch.norm(rel_no_time, p=4),
            torch.norm(rhs, p=4),
            torch.norm(self.U, p=2),
            torch.norm(self.V, p=2),
        ), self.T.weight)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.E.weight.data[chunk_begin : chunk_begin + chunk_size].transpose(
            0, 1
        )

    def get_queries(self, queries: torch.Tensor):
        lhs = self.E(queries[:, 0])
        rel = self.R(queries[:, 1])
        rel_no_time = self.R_noT(queries[:, 1])
        time = self.T(queries[:, 3])

        rel_t = rel * time + rel_no_time

        lhs = self.bn0(lhs)
        lhs = self.input_dropout(lhs)

        x = torch.mm(lhs, self.U) * torch.mm(rel_t, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.m(x)

        return x
