from typing import Tuple
import torch


class MOTPLoss(torch.nn.Module):
    def __init__(self, accuracy: float = 1.0):
        super(MOTPLoss, self).__init__()

        self.accuracy = accuracy

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distance = torch.norm((prediction - target), dim=1)


        return (torch.sum(distance) / distance.shape[0],
            torch.sum(distance < self.accuracy) / distance.shape[0]
        )
        