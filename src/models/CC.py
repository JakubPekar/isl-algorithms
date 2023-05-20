import torch
import torch.nn
import torch.fft
import math
from typing import Optional
from multipledispatch import dispatch



class CC(torch.nn.Module):
    def __init__(
            self,
            sampling_rate: int,
            max_tau: Optional[int] = None,
            tdoa_estimate: bool = False,
        ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.max_tau = max_tau
        self.tdoa_estimate = tdoa_estimate



    @dispatch(torch.Tensor, torch.Tensor)
    def forward( # type: ignore 
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        n = x.shape[-1] + y.shape[-1]

        # Cross Correlation In Frequency Domain using convolutional theorem
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        cc = torch.fft.fftshift(torch.fft.irfft(Gxy, n), dim=-1)


        if self.max_tau and self.max_tau < n / 2:
            cc = cc[..., n // 2 - self.max_tau : n // 2 + self.max_tau + 1]

        # Return TDOA
        if self.tdoa_estimate:
            return torch.argmax(cc, dim=-1) \
                - math.floor(cc.shape[-1] / 2)

        return cc



    @dispatch(torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(*torch.split(x, 1, dim=-2))
