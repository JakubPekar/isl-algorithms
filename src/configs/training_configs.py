import random
import torch
import torch.utils.data
import torch.nn.functional as F
from typing import Callable, Dict, List
from src.losses.LabelSmoothing import LabelSmoothing
from src.RealWorldDataset import RealWorldDataset
from src.SyntheticDataset import SyntheticDataset
import src.models as models
from src.utils.types import TrainingConfig, TrainingDatasetConfig
from src.utils.utils import max_tau, true_tdoa, window_size
from src.utils.constants import FINE_TUNE_LABELS_PATH, R_RECEIVERS, SAMPLING_FREQUENCY, TRAINING_LABELS_PATH
from src.transforms import RandomCrop, Compose, ToTensor



MAX_TAU = [
    max_tau(R_RECEIVERS, SAMPLING_FREQUENCY),
    max_tau(R_RECEIVERS, SAMPLING_FREQUENCY, diag=True),
    max_tau(R_RECEIVERS, SAMPLING_FREQUENCY, diag=True),
]


def get_tdoa_choices(key: int) -> List[List[int]]:
    return [
        # Vertical and horizontal
        [[0, 2], [1, 3]],
        # Diagonal 1
        [[0, 1], [2, 3]],
        # Diagonal 2
        [[0, 3], [1, 2]],
    ][key]
        

def get_tdoa_training_receivers_extractor(
    key: int
) -> Callable[[List[int]], List[int]]:
    return lambda receivers: random.choice(get_tdoa_choices(key))



def training_datasets(
    config: TrainingDatasetConfig, fine: bool
) -> torch.utils.data.Dataset:
    if fine:
        return RealWorldDataset(
            FINE_TUNE_LABELS_PATH,
            SAMPLING_FREQUENCY,
            gt_extractor=config['gt_extractor'],
            receiver_extractor=config['receivers_extractor'], # type: ignore
            transform=Compose([
                ToTensor(),
                RandomCrop(
                    config['n_channels'],
                    window_size(SAMPLING_FREQUENCY)
                ),
            ]),
        )
    
    return SyntheticDataset(
        TRAINING_LABELS_PATH,
        SAMPLING_FREQUENCY,
        gt_extractor=config['gt_extractor'],
        receiver_extractor=config['receivers_extractor'], # type: ignore
        transform=Compose([
            ToTensor(),
            RandomCrop(
                config['n_channels'],
                window_size(SAMPLING_FREQUENCY)
            ),
        ]),
    )




def pgcc_phat_config(key: int) -> Callable[[torch.device], TrainingConfig]:
    def get_config(device: torch.device) -> TrainingConfig:
        model = models.PGCC_PHAT(device, max_tau=MAX_TAU[key])

        return model, {
            'name': 'pgcc-phat',
            'dataset_config': {
                'n_channels': 2,
                'receivers_extractor':
                    get_tdoa_training_receivers_extractor(key),
                'gt_extractor': lambda s, recs: true_tdoa(s, recs) \
                    / MAX_TAU[key],
            },
            'criterion': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam(model.parameters()),
            'epochs': 80,
            'batch_size': 128,
        }
    
    return get_config


def ngcc_phat_config(key: int) -> Callable[[torch.device], TrainingConfig]:
    def get_config(device: torch.device) -> TrainingConfig:
        model = models.NGCC_PHAT(
            device, max_tau=MAX_TAU[key], sampling_rate=SAMPLING_FREQUENCY)
        
        return model, {
            'name': 'ngcc-phat',
            'dataset_config': {
                'n_channels': 2,
                'receivers_extractor':
                    get_tdoa_training_receivers_extractor(key),
                'gt_extractor': lambda s, recs: MAX_TAU[key] \
                    + torch.round(true_tdoa(s, recs)),
            },
            'criterion': LabelSmoothing(),
            'optimizer': torch.optim.Adam(model.parameters()),
            'epochs': 50,
            'batch_size': 32,
        }
    
    return get_config


def e2e_cnn_config(device: torch.device) -> TrainingConfig:
    model = models.E2E_CNN(4, window_size(SAMPLING_FREQUENCY))

    return model, {
        'name': 'e2e-cnn',
        'dataset_config': {
            'n_channels': 4,
            'receivers_extractor': lambda x: x, # Identity
            'gt_extractor': lambda x, _: x, # Identity
        },
        'criterion': torch.nn.MSELoss(),
        'optimizer': torch.optim.Adam(model.parameters()),
        'epochs': 200,
        'batch_size': 100,
    }



def get_training_configs(
) -> Dict[str, Callable[[torch.device],TrainingConfig]]:
    return {
        # Because PGCC-PHAT and NGCC-PHAT are TDOA extractors
        'pgcc-phat-0': pgcc_phat_config(0),
        'pgcc-phat-1': pgcc_phat_config(1),
        'pgcc-phat-2': pgcc_phat_config(2),
        'ngcc-phat-0': ngcc_phat_config(0),
        'ngcc-phat-1': ngcc_phat_config(1),
        'ngcc-phat-2': ngcc_phat_config(2),
        'e2e-cnn': e2e_cnn_config,
    }
