import math
import soundfile
import torch
import random
import numpy as np
from torch import Tensor
from typing import Callable, List, Optional, Tuple
from src.utils.types import Coordinates, Sample, Signal
from src.utils.constants import SAMPLING_FREQUENCY, SOUND_VELOCITY, WINDOW_SIZE



time_rates = {
    1 / 60**2: 'h',
    1 / 60: 'min',
    1: 's',
    1e3: 'ms',
    1e6: 'µs',
    1e9: 'ns',
}



def manual_seed(v: int = 0):
    torch.manual_seed(v)
    random.seed(v)
    np.random.seed(v)


def circular_array(n: int, r: float) -> Tensor:
    return torch.round(torch.tensor([
        [
          0,
          r * math.cos(2 * math.pi * i / n),
          r * math.sin(2 * math.pi * i / n)
        ] for i in range(n)]),
        decimals=2
    )



def process_signal(sample: Sample, f: Callable[[Tensor], Tensor]) -> Sample:
    return f(sample[0]), sample[1]



def euclid_distance(a: Tensor, b: Tensor) -> Tensor:
    return torch.norm(a - b, dim=-1)



def true_tdoa(
    s: Tensor,
    recs: Tensor,
    samplerate: Optional[int] = SAMPLING_FREQUENCY
) -> Tensor:
    # - Because diff is computed as s[i+1] - s[i]
    return -torch.diff(euclid_distance(
        # Respahe the source tensor
        s.view(
          *s.shape[:-1],
          *[1] * len(recs.shape[:-1]),
          s.shape[-1]
        ).expand(
          *s.shape[:-1],
          *recs.shape[:-1],
          s.shape[-1]
        ),
        recs
    ), dim=-1) / SOUND_VELOCITY * samplerate



def shift_signals(
    device: torch.device, s1: Tensor, s2: Tensor, tau: Tensor
) -> Tuple[Tensor, Tensor]:
    tau = tau.unsqueeze(-1)
    mask = torch.ones(s1.shape, dtype=torch.bool).to(device)
    mask_range = torch.arange(0, s1.shape[-1]).expand(
        *s1.shape[:-1], s1.shape[-1]
    ).to(device)

    mask[mask_range >= s1.shape[-1] - tau] = 0
    mask[mask_range <= -tau] = 0

    return s1 * mask, s2 * mask.flip([-1])



# Compute the centroid of given receivers (average of each dimension)
def relative_receivers(recs: List[Coordinates]) -> np.ndarray:
    return np.round(recs - np.mean(recs, axis=-2), decimals=2)



def signal_energy(signal: Tensor) -> Tensor:
    return torch.sum(signal ** 2, dim=-1)



def tdoa_to_distance(tdoa: Tensor, samplerate: int) -> Tensor:
    return tdoa / samplerate * SOUND_VELOCITY



def max_tau(radius: float, samplerate: int, diag: bool = False) -> int:
    if diag:
        return math.ceil(
            math.sqrt(2 * radius ** 2) / SOUND_VELOCITY * samplerate)
  
    return math.ceil(2 * radius / SOUND_VELOCITY * samplerate)


def window_size(samplerate: int) -> int:
    return math.ceil(WINDOW_SIZE * samplerate)



def load_signal(path: str, samplerate: Optional[int] = None) -> Signal:
    s, fs = soundfile.read(path)
    s = s.T

    if samplerate is None:
        return s
    
    if samplerate > fs:
        raise Exception('The sampling rate is too high')

    return s[..., ::fs // samplerate]




combinations = torch.combinations(
    torch.tensor(range(4)), 2)[[[1, 4], [0, 5], [2, 3]], ...]


def split_signal_tdoa(signal: Tensor) -> List[Tensor]:
    s1, s2, s3 = torch.split(signal[..., combinations, :], 1, dim=-4)
    orig = s1.shape
    return [
        s1.reshape(-1, *orig[-2:]),
        s2.reshape(-1, *orig[-2:]),
        s3.reshape(-1, *orig[-2:]),
        orig # type: ignore
    ] 


def merge_signal_tdoa(
    s1: Tensor, s2: Tensor, s3: Tensor, shape: List[int]
) -> Tensor:
    
    return torch.stack([
         s2.reshape(*shape[:-2], 1)[..., 0, :],
         s1.reshape(*shape[:-2], 1)[..., 0, :],
         s3.reshape(*shape[:-2], 1)[..., 0, :],
         s3.reshape(*shape[:-2], 1)[..., 1, :],
         s1.reshape(*shape[:-2], 1)[..., 1, :],
         s2.reshape(*shape[:-2], 1)[..., 1, :],
    ], dim=-2).squeeze(-3)


def get_time(time: float) -> str:
    for time_rate, unit in time_rates.items():
        if time * time_rate >= 1:
            return f'{round(time * time_rate, 2)}{unit}'

    return f'{time * list(time_rates)[-1]}{time_rates[list(time_rates)[-1]]}'
