import torch as th
import torch.nn as nn

from marl_classification.networks.permute import Permute


class MessageSender(nn.Module):
    """
    m_θ4 : R^n -> R^n_m
    """

    def __init__(self, n: int, n_m: int,
                 hidden_size: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m
        self.__n_e = hidden_size

        self.__seq_lin = nn.Sequential(
			nn.Linear(self.__n, self.__n_e),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(self.__n_e),
            Permute([2, 0, 1]),
            nn.Linear(self.__n_e, self.__n_m),
        )

        for m in self.__seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_t: th.Tensor) -> th.Tensor:
        return self.__seq_lin(h_t)