import torch
import os
import torch.utils.data
import numpy as np
from typing import List
from src.losses.RMSELoss import RMSELoss
from src.Algorithms import Algorithms
from src.SyntheticDataset import SyntheticDataset
from src.transforms import Compose, ToTensor
from src.transforms import OverlapSplit
from src.utils.constants import SAMPLING_FREQUENCY, TESTING_PATH
from src.utils.ProgressBar import ProgressBar
from src.utils.utils import true_tdoa


@torch.no_grad()
def evaluate_tdoa_synthetic(
    algorithms: Algorithms,
    models: List[str],
    device: torch.device,
):
    results = {}

    # datasets = list(filter(
    #     lambda file: file.endswith('.json'),
    #     os.listdir(TESTING_PATH)
    # ))

    combinations = torch.combinations(torch.tensor(range(4)), 2)

    datasets = [
        f'SNR_{snr}.json' for snr in np.linspace(-10, 30, 9)
    ] + [
        f'RT60_{rt60}.json' for rt60 in np.linspace(0.2, 1, 9)
    ]

    for i, dataset_name in enumerate(datasets, start=1):
        print(f'Evaluating tdoa ({i}/{len(datasets)}')

        # initialize results
        results[dataset_name] = {}
        for model in models:
            results[dataset_name][model] = 0

        criterion = RMSELoss()

        dataset = SyntheticDataset(
            os.path.join(TESTING_PATH, dataset_name),
            SAMPLING_FREQUENCY,
            lambda s, recs: torch.round(true_tdoa(s, recs[combinations])),
            transform=Compose([
                ToTensor(),
                OverlapSplit()
            ]),
        )

        test_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, # Must remain 1 because of OverlapSplit
            shuffle=False,
            drop_last=False,
        )

        logger = ProgressBar(len(test_data_loader), 1)

        for step, (samples, targets) in enumerate(test_data_loader, start=1):
            samples = samples.to(device, non_blocking=True).squeeze(0)
            targets = targets.to(device, non_blocking=True).squeeze(0)

            for model in models:
                output = getattr(algorithms, f'{model}_tdoa')(samples)

                loss = criterion(output, targets)

                results[dataset_name][model] += loss.item() \
                    / len(test_data_loader)

            logger(step)
    
    return results
