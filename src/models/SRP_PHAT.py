from typing import List
import torch
import torch.nn
import torch.fft
from src.models.GCC import GCC
from src.utils.types import Coordinates
from src.utils.utils import true_tdoa
from src.utils.constants import MAX_SPEAKER_HEIGHT, MIC_DISTANCE_FROM_WALL, MIN_SPEAKER_HEIGHT



class SRP_PHAT(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        sampling_rate: int,
        max_tau: int,
        density: List[float] = [0.1, 0.1, 0.1],
        relative: bool = True
    ):
        super().__init__()

        gcc = GCC(device, max_tau=max_tau)

        self.device = device
        self.density = density
        self.max_tau = max_tau
        self.sampling_rate = sampling_rate
        self.relative = relative

        # To be initialized
        self.x: torch.Tensor
        self.tau: torch.Tensor
        self.combinations: torch.Tensor
        self.rec_centroid: torch.Tensor



        R = lambda s: torch.gather(
            gcc(s[..., self.combinations, :]).view(
                *s.shape[:-2],
                *[1] * len(self.tau.shape[:-2]),
                self.tau.shape[-2],
                2 * self.max_tau + 1
            ).expand(
                *s.shape[:-2],
                *self.tau.shape[:-1],
                2 * self.max_tau + 1
            ),
            -1,
            self.tau.expand(*s.shape[:-2], *self.tau.shape)
        ).squeeze(-1)

        self.P = lambda s: torch.sum(R(s), dim=-1)
  


    def init(
        self, receivers: torch.Tensor, room_dimensions: Coordinates
    ) -> None:
        self.rec_centroid = torch.mean(receivers, dim=-2)
        
        self.combinations = torch.combinations(
            torch.tensor(range(len(receivers))), 2).to(self.device)
        
        self.x = torch.tensor([
            [x, y, z] for x in torch.arange(
                MIC_DISTANCE_FROM_WALL, room_dimensions[0] + self.density[0], self.density[0]
            ) for y in torch.arange(
                0, room_dimensions[1] + self.density[1], self.density[1]
            ) for z in torch.arange(
                MIN_SPEAKER_HEIGHT, MAX_SPEAKER_HEIGHT + self.density[2], self.density[2]
            )
        ]).flatten(0, -2).to(self.device)
    

        self.tau = self.max_tau + torch.round(true_tdoa(
            self.x, receivers[self.combinations], self.sampling_rate
        )).to(torch.long)
  


    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.x[torch.argmax(self.P(signal), dim=-1)]
        return x - self.rec_centroid if self.relative else x
