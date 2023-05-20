import torch
import torch.nn
import torch.fft
import math
from typing import Optional
from multipledispatch import dispatch



class GCC(torch.nn.Module):
    def __init__(
            self,
            device: torch.device,
            max_tau: Optional[int] = None,
            epsilon: float = 0.001,
            beta: torch.Tensor = torch.tensor([1]),
            tdoa_estimate: bool = False,
        ):
        super().__init__()

        self.max_tau = max_tau
        self.epsilon = epsilon
        self.beta = beta.unsqueeze(1).to(device)
        self.tdoa_estimate = tdoa_estimate



    @dispatch(torch.Tensor, torch.Tensor)
    def forward( # type: ignore 
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        n = x.shape[-1] + y.shape[-1]

        # Generalized Cross Correlation Phase Transform
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        # Epislon to avoid division by zero
        phi = 1 / (torch.abs(Gxy) + self.epsilon)

        gcc = torch.fft.fftshift(torch.fft.irfft(
            Gxy * torch.pow(phi, self.beta), n
        ), dim=-1)

        if self.max_tau and self.max_tau < n / 2:
            gcc = gcc[..., n // 2 - self.max_tau : n // 2 + self.max_tau + 1]


        # Return TDOA
        if self.tdoa_estimate:
            return torch.argmax(gcc, dim=-1) \
                - math.floor(gcc.shape[-1] / 2)

        return gcc



    @dispatch(torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(*torch.split(x, 1, dim=-2))
