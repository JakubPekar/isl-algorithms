import numpy as np
from src.utils.types import Sample
from src.utils.utils import process_signal



class Normalize:
    def __init__(self) -> None:
        self.f = np.vectorize(lambda x: x / max(abs(x)), signature='(n)->(n)')

    def __call__(self, sample: Sample) -> Sample:
        return process_signal(sample, self.f)
