import json
import torch
import torch.utils.data
import numpy as np
from typing import Callable, List, Optional, Union
from src.utils.types import GtExtractor, Sample
from src.utils.utils import load_signal



class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        sample_rate: int,
        gt_extractor: GtExtractor = lambda x, _: x,
        receiver_extractor: Callable[
            [Union[List[int], torch.Tensor]],
            Union[List[int], torch.Tensor]
        ] = torch.nn.Identity(),
        transform: Optional[Callable[[Sample], Sample]] = None,
    ) -> None:
        super(SyntheticDataset, self).__init__()
        self.transform = transform
        self.receiver_extractor = receiver_extractor
        self.gt_extractor = gt_extractor
        self.sample_rate = sample_rate

        with open(data_path, 'r') as f:
            self.data = json.load(f)
        


    def __len__(self) -> int:
        return len(self.data)



    def __getitem__(self, index: int) -> Sample:
        sample = self.data[index]

        # Translate to relative coordinates
        receivers = np.array(sample['receivers'])
        receivers_centroid = np.mean(receivers, axis=0)
        receivers -= receivers_centroid
        
        # Filter microphones
        mics_indexes = self.receiver_extractor(list(
            range(len(sample['receivers']))
        ))

        X = torch.from_numpy(np.take(load_signal(
            sample['signal'], self.sample_rate
        ), mics_indexes, axis=0))
        
        y = self.gt_extractor(
            torch.tensor(sample['source_position']) \
                - torch.from_numpy(receivers_centroid),
            torch.from_numpy(np.take(sample['receivers'], mics_indexes, axis=0))
        )

        if self.transform is not None:
            return self.transform((X, y))

        return X, y
