import torch
from torch import nn
from multipledispatch import dispatch
from src.models.SincNet import SincNet
from src.models.GCC import GCC
from src.utils.utils import window_size



class NGCC_PHAT(nn.Module):
    def __init__(
            self,
            device: torch.device,
            max_tau: int,
            sampling_rate: int,
        ):
        super().__init__()

        self.max_tau = max_tau

        sincnet_params = {
            'input_dim': window_size(sampling_rate),
            'fs': sampling_rate,
            'cnn_N_filt': [128, 128, 128, 128],
            'cnn_len_filt': [1023, 11, 9, 7],
            'cnn_max_pool_len': [1, 1, 1, 1],
            'cnn_use_laynorm_inp': False,
            'cnn_use_batchnorm_inp': False,
            'cnn_use_laynorm': [False, False, False, False],
            'cnn_use_batchnorm': [True, True, True, True],
            'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
            'cnn_drop': [0.0, 0.0, 0.0, 0.0],
            'use_sinc': True,
        }

        block = lambda k: nn.Sequential(
            nn.Conv1d(128, 128, k, padding=k//2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.backbone = SincNet(sincnet_params)
        self.gcc = GCC(device, max_tau=max_tau)

        self.block1 = block(11)
        self.block2 = block(9)
        self.block3 = block(7)

        self.covn_out = nn.Conv1d(128, 1, 5, padding=2)



    @dispatch(torch.Tensor, torch.Tensor)
    def forward( # type: ignore
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Reshape the input to (batch_size, 2, tau) for TDOA evaluation
        # Reshape it to the original shape after TDOA estimate
        orig_shape = x.shape
        x = x.view(-1, *orig_shape[-2:])
        y = y.view(-1, *orig_shape[-2:])

        x = self.backbone(x)
        y = self.backbone(y)

        x = self.gcc(x, y)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.covn_out(x)

        return x.view(*orig_shape[:-2], x.shape[-1])
    


    @dispatch(torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(*torch.split(x, 1, dim=-2))
    

    def get_max_tau(self) -> int:
        return self.max_tau
