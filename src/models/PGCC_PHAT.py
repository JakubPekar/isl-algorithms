import torch
import torch.nn.functional as F
from torch import nn
from multipledispatch import dispatch
from src.models.GCC import GCC



class PGCC_PHAT(nn.Module):
    def __init__(
            self,
            device: torch.device,
            max_tau: int,
            beta: torch.Tensor = torch.linspace(0, 1, 11), 
        ):
        super().__init__()

        self.max_tau = max_tau
        self.gcc = GCC(device, max_tau=max_tau, beta=beta)

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * (2 * max_tau - 9) * (len(beta) - 10), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        self.drop1 = nn.Dropout1d(0.2)
        self.drop2 = nn.Dropout1d(0.2)

        self.fbn1 = nn.BatchNorm1d(512)
        self.fbn2 = nn.BatchNorm1d(512)


    @dispatch(torch.Tensor, torch.Tensor)
    def forward( # type: ignore
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Reshape the input to (batch_size, 2, tau) for TDOA evaluation
        # Reshape it to the original shape after TDOA estimate
        orig_shape = x.shape
        x = x.view(-1, *orig_shape[-2:])
        y = y.view(-1, *orig_shape[-2:])
        
        with torch.no_grad():
            x = self.gcc(x, y)

        # reshape to (batch_size, 1, beta, tau)
        x = torch.unsqueeze(x, dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.drop1(F.relu(self.fbn1(self.fc1(x))))
        x = self.drop2(F.relu(self.fbn2(self.fc2(x))))
        x = self.fc3(x)
        return x.view(*orig_shape[:-2], 1)
    


    @dispatch(torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(*torch.split(x, 1, dim=-2))


    def normalization(self) -> float:
        return self.max_tau