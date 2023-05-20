import torch
from src.utils.types import Sample


class ToTensor:
    def __call__(self, sample: Sample) -> Sample:
        return sample[0].to(torch.float), sample[1].to(torch.float)
