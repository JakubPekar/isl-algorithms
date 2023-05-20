import torch
import torch.nn
from src.utils.utils import shift_signals, signal_energy



class ILD(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device


    def forward( # type: ignore
        self, s1: torch.Tensor, s2: torch.Tensor, tau_12: torch.Tensor
    ) -> torch.Tensor:
        s1, s2 = shift_signals(self.device, s1, s2, tau_12)
        E1 = signal_energy(s1)
        E2 = signal_energy(s2)
        return E1 / E2
