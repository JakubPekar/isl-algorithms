import torch
import numpy as np
import numpy.typing as npt
from typing import Annotated, Callable, Literal, Optional, Tuple, TypedDict, Union, List



Signal = Annotated[npt.NDArray[np.floating], Literal['N']]
Coordinates = Union[List[float], List[int]]
Sample = Tuple[torch.Tensor, torch.Tensor]
GtExtractor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EvalPipeline = Callable[[torch.Tensor], torch.Tensor]

DatasetInfo = TypedDict(
    'DatasetInfo',
    {
        'receivers': torch.Tensor,
        'room-dimensions': List[float],
    }
)

PartialRoomSimulationConfig = TypedDict(
    'PartialRoomSimulationConfig',
    {
        'room_dimensions': Coordinates,
        'receivers': List[Coordinates],
        'signal': Tuple[Signal, int],
    }
)

RoomSimulationConfig = TypedDict(
    'RoomSimulationConfig',
    {
        'room_dimensions': Coordinates,
        'source_position': Coordinates,
        'receivers': List[Coordinates],
        'signal': Tuple[Signal, int],
        'rt60': Optional[float],
        'snr': Optional[int],
    }
)

TrainingDatasetConfig = TypedDict(
    'TrainingDatasetConfig',
    {
        'n_channels': int,
        'receivers_extractor': Callable[[List[int]], List[int]],
        'gt_extractor': GtExtractor,
    }
)



TrainingParams = TypedDict(
    'TrainingParams',
    {
        'name': str,
        'dataset_config': TrainingDatasetConfig,
        'optimizer': torch.optim.Optimizer,
        'criterion': torch.nn.Module,
        'epochs': int,
        'batch_size': int,
    },
)

TrainingConfig = Tuple[torch.nn.Module, TrainingParams]

RoomSimulationConfigGenerator = Callable[[str], RoomSimulationConfig]