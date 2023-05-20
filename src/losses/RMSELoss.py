import torch


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self.mse(prediction, target))
        