import torch
import os
import numpy as np
import torch.utils.data
from typing import List
from src.Algorithms import Algorithms
from src.SyntheticDataset import SyntheticDataset
from src.losses.MOTPLoss import MOTPLoss
from src.transforms import Compose, ToTensor
from src.transforms import OverlapSplit
from src.utils.constants import R_RECEIVERS, SAMPLING_FREQUENCY, TESTING_PATH, TESTING_ROOM_DIMENSIONS
from src.utils.generators import circular_array_placement
from src.utils.ProgressBar import ProgressBar


@torch.no_grad()
def evaluate_synthetic(
    algorithms: Algorithms,
    models: List[str],
    device: torch.device,
):
    results = {}

    # datasets = list(filter(
    #     lambda file: file.endswith('.json'),
    #     os.listdir(TESTING_PATH)
    # ))

    datasets = [
        f'SNR_{snr}.json' for snr in np.linspace(-10, 30, 9)
    ] + [
        f'RT60_{rt60}.json' for rt60 in np.linspace(0.2, 1, 9)
    ]

    for i, dataset_name in enumerate(datasets, start=1):
        print(f'Evaluating ({i}/{len(datasets)}')

        # initialize results
        results[dataset_name] = {}
        for model in models:
            results[dataset_name][model] = { 'MOTP': 0, 'Accuracy': 0 }

        criterion = MOTPLoss()

        dataset = SyntheticDataset(
            os.path.join(TESTING_PATH, dataset_name),
            SAMPLING_FREQUENCY,
            transform=Compose([
                ToTensor(),
                OverlapSplit()
            ]),
        )

        algorithms.init_data({
            'room-dimensions': TESTING_ROOM_DIMENSIONS, # type: ignore
            'receivers': circular_array_placement(
                TESTING_ROOM_DIMENSIONS, R_RECEIVERS
            ),
        })

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
                output = getattr(algorithms, model)(samples)

                mopt_loss, accuracy = criterion(output, targets)

                results[dataset_name][model]['MOTP'] += mopt_loss.item() \
                    / len(test_data_loader)
                
                results[dataset_name][model]['Accuracy'] += accuracy.item() \
                    / len(test_data_loader)

            logger(step)
    
    return results
