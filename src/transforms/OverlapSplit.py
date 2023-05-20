from src.utils.types import Sample
from src.utils.utils import process_signal
from src.utils.constants import SAMPLING_FREQUENCY, WINDOW_SIZE
import torchvision.transforms as T


class OverlapSplit:
    def __init__(self) -> None:
        self.split_each = SAMPLING_FREQUENCY * WINDOW_SIZE

        self.f = lambda x: x.T.unfold(
            dimension=0, size=self.split_each, step=self.split_each // 2)
    

    def __call__(self, sample: Sample) -> Sample:
        if sample[0].shape[-1] < self.split_each:
            X, y = process_signal(
                sample,
                T.RandomCrop(
                    (sample[0].shape[-2], self.split_each),
                    pad_if_needed=True
                )
            )

            return X.unsqueeze(0), y.unsqueeze(0)
      
        
        X, y = process_signal(sample, self.f)
        return X, y.expand(X.shape[0], *y.shape)
