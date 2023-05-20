import torch
import torch.utils.data
from typing import List
from src.Algorithms import Algorithms
from src.losses.RMSELoss import RMSELoss
from src.RealWorldDataset import RealWorldDataset
from src.transforms import Compose, ToTensor
from src.transforms import OverlapSplit
from src.utils.constants import SAMPLING_FREQUENCY
from src.utils.ProgressBar import ProgressBar
from src.utils.utils import true_tdoa


@torch.no_grad()
def evaluate_tdoa_real(
    algorithms: Algorithms,
    models: List[str],
    datasets: List[str],
    device: torch.device,
):
    results = {}

    combinations = torch.combinations(torch.tensor(range(4)), 2)

    for dataset_path in datasets:
        print(f'Evaluating toda on {dataset_path}')

        # initialize results
        results[dataset_path] = {}
        for model in models:
            results[dataset_path][model] = 0

        criterion = RMSELoss()

        dataset = RealWorldDataset(
            dataset_path,
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

                results[dataset_path][model] += loss.item() \
                     / len(test_data_loader)

            logger(step)
    
    return results
