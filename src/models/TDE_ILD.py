import torch
import torch.nn
import torch.fft
from src.models import GCC, ILD
from src.utils.utils import tdoa_to_distance



class TDE_ILD(torch.nn.Module):
    def __init__(
            self,
            device: torch.device,
            max_tau: int,
            sampling_rate: int,
            R: float,
        ):
        super().__init__()

        self.device = device
        self.R = R
        self.max_tau = max_tau
        self.sampling_rate = sampling_rate
        self.gcc = GCC(device, max_tau, tdoa_estimate=True)
        self.ild = ILD(device)



    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        s1, s2, s3, s4 = torch.split(signal, 1, dim=-2)
        tau_13 = self.gcc(s1, s3) # Horizontal
        tau_24 = self.gcc(s2, s4) # Vertical

        m = torch.sqrt(self.ild(s1, s3, -tau_13))

        d_13 = tdoa_to_distance(tau_13, self.sampling_rate)
        d_24 = tdoa_to_distance(tau_24, self.sampling_rate)

        r_1 = d_13 / (1 - m)
        r_2 = d_13 * m / (1 - m)

        # Receivers positioned on the y-axis
        y = (r_2 ** 2 - r_1 ** 2) / (4 * self.R)
        x = torch.sqrt(r_1 ** 2 - (y - self.R) ** 2)

        r = torch.norm(torch.stack([x, y], dim=-1), dim=-1)
        
        phi = torch.acos(d_13 / (2 * self.R))
        theta = torch.acos(d_24 / (2 * self.R))

        # [x, y, z]
        return torch.nan_to_num(torch.stack([
            r * torch.sin(phi) * torch.sin(theta),
            - r * torch.cos(phi) * torch.sin(theta),
            - r * torch.cos(theta)
        ], dim=-1).squeeze(-2))
