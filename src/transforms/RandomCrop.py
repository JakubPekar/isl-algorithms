import torchvision.transforms as T
from src.utils.types import Sample
from src.utils.utils import process_signal



class RandomCrop:
    def __init__(self, in_dim: int, size: int) -> None:
        self.f = T.RandomCrop((in_dim, size), pad_if_needed=True)

    def __call__(self, sample: Sample) -> Sample:
        return process_signal(sample, self.f)
