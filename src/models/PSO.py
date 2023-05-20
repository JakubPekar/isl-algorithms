import torch
import pyswarms as ps
import numpy as np
from src.losses.PSOLoss import PSO_Loss
from src.utils.types import Coordinates
from src.utils.constants import MAX_SPEAKER_HEIGHT, MIC_DISTANCE_FROM_WALL, MIN_SPEAKER_HEIGHT, PLACEMENT


# PSO parameters
N_PARTICLES = 20
N_ITERATIONS = 50

class PSO(torch.nn.Module):
    def __init__(
            self,
            device: torch.device,
            sampling_rate: int,
            relative: bool = True
        ):
        super().__init__()

        self.device = device
        self.relative = relative
        self.sampling_rate = sampling_rate
        self.pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # To be initialized
        self.room_dimensions: Coordinates
        self.rec_centroid: torch.Tensor
        
        self.pso_loss: PSO_Loss

        def pso(tdoa):
            return self.get_pso_optimizer(
                torch.tensor([
                    self.rec_centroid[0] + 1,
                    self.rec_centroid[1],
                    self.rec_centroid[2],
                ]).cpu() if PLACEMENT == 'ceiling' else self.rec_centroid.cpu(),
            ).optimize(
                lambda x: self.pso_loss(tdoa)(torch.from_numpy(x)).numpy(),
                iters=N_ITERATIONS,
                verbose=False
            )[1]

        self.pso_v = np.vectorize(pso, signature='(n, m)->(3)')



    def init(
        self, receivers: torch.Tensor, room_dimensions: Coordinates
    ) -> None:
        combinations = torch.combinations(
            torch.tensor(range(len(receivers))), 2)
        
        self.rec_centroid = torch.mean(receivers, dim=-2)
        self.room_dimensions = room_dimensions

        self.pso_loss = PSO_Loss(
            receivers[combinations].cpu(),
            self.sampling_rate
        )



    def get_pso_optimizer(
        self, init_position: torch.Tensor
    ) -> ps.single.GlobalBestPSO:
        return ps.single.GlobalBestPSO(
            n_particles=N_PARTICLES,
            dimensions=3,
            options=self.pso_options,
            init_pos=init_position.expand(
                N_PARTICLES, *init_position.shape).numpy(),
            # min speaker height
            bounds=([MIC_DISTANCE_FROM_WALL, 0, MIN_SPEAKER_HEIGHT],
                    self.room_dimensions[:2] + [MAX_SPEAKER_HEIGHT])
        )



    def forward(self, tdoas: torch.Tensor) -> torch.Tensor:
        orig_shape = tdoas.shape
        tdoas = tdoas.view(-1, *orig_shape[-2:])
        
        x = torch.from_numpy(self.pso_v(tdoas.cpu())).view(*orig_shape[:-2], 3)\
            .to(self.device)

        return x - self.rec_centroid if self.relative else x
