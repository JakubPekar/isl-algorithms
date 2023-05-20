import torch
import torch.nn.functional as F
from src.utils.utils import tdoa_to_distance
from src.utils.types import Coordinates



class LS(torch.nn.Module):
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

        # To be initialized
        self.receivers: torch.Tensor
        self.room_dimensions: torch.Tensor
        self.A: torch.Tensor
        self.b: torch.Tensor
        self.i: torch.Tensor
        self.reference_receiver: torch.Tensor
        self.rec_centroid: torch.Tensor



    def init(
        self, receivers: torch.Tensor, room_dimensions: Coordinates
    ) -> None:
        combinations = torch.combinations(
            torch.tensor(range(len(receivers))), 2)
        
        self.rec_centroid = torch.mean(receivers, dim=-2)
        receivers = receivers - self.rec_centroid

        self.receivers = receivers[combinations]
        self.room_dimensions = torch.tensor(room_dimensions).to(self.device)

        self.A = torch.diff(self.receivers, dim =-2).squeeze(-2)
        self.b = torch.diff(torch.sum(self.receivers ** 2, dim=-1)) \
            .squeeze(-1)
        self.i = combinations[:, 0]
        self.reference_receiver = receivers[0]



    # Predictions of LS are commonly totally off...
    def prediction_clamp(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.nan_to_num(prediction) + self.rec_centroid,
            min=torch.zeros(3).to(self.device),
            max=self.room_dimensions
        ) - (self.rec_centroid if self.relative else 0)
        


    def forward(self, tdoas: torch.Tensor) -> torch.Tensor:
        distances = tdoa_to_distance(tdoas, self.sampling_rate)
        
        A = 2 * torch.cat([
            self.A.expand(distances.shape[0], *self.A.shape),
            distances
        ], dim=-1)

        dist_tdoas = distances.squeeze(-1)

        b = self.b - dist_tdoas ** 2 - 2 * dist_tdoas * \
            F.pad(dist_tdoas[..., :3], (1, 0))[..., self.i]

        # Least Squares on batch
        ls = torch.bmm(torch.pinverse(A), b.unsqueeze(-1)).squeeze()

        # Quadratic equation to obtain the x coordinate
        # In the followinf code I work with the prerequisite that coordinate of reference receiver on the x axis is 0 -> ommiting the b factor in quadratice equation
        x = torch.sqrt(ls[..., 3] ** 2 - torch.sum((ls[..., :3] - self.reference_receiver) ** 2, dim=-1))

        estimate = torch.cat((x.unsqueeze(-1), ls[..., 1:3]), dim=-1)
        return self.prediction_clamp(estimate)
