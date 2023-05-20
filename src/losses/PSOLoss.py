from typing import Callable
import torch
from src.utils.utils import true_tdoa



class PSO_Loss(torch.nn.Module):
    def __init__(self, receivers: torch.Tensor, sampling_rate: int):
        super(PSO_Loss, self).__init__()

        self.receivers = receivers
        self.sampling_rate = sampling_rate

        # To be initialized
        self.tdoas: torch.Tensor
      


    def forward(
        self, tdoas: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x: torch.sum((torch.round(true_tdoa(
            x, self.receivers, self.sampling_rate
        )) - tdoas) ** 2, dim=-2).squeeze(-1)
