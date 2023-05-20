import torch
import torch.utils.data
from typing import List
from src.Algorithms import Algorithms
from src.RealWorldDataset import RealWorldDataset
from src.losses.MOTPLoss import MOTPLoss
from src.transforms import Compose, ToTensor
from src.transforms import OverlapSplit
from src.utils.constants import SAMPLING_FREQUENCY
from src.utils.ProgressBar import ProgressBar


@torch.no_grad()
def evaluate_real(
    algorithms: Algorithms,
    models: List[str],
    datasets: List[str],
    device: torch.device,
):
    results = {}

    for dataset_path in datasets:
        print(f'Evaluating {dataset_path}')

        # initialize results
        results[dataset_path] = {}
        for model in models:
            results[dataset_path][model] = { 'MOTP': 0, 'Accuracy': 0 }

        criterion = MOTPLoss()

        dataset = RealWorldDataset(
            dataset_path,
            SAMPLING_FREQUENCY,
            transform=Compose([
                ToTensor(),
                OverlapSplit()
            ]),
        )

        algorithms.init_data(dataset.info())

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

                results[dataset_path][model]['MOTP'] += mopt_loss.item() \
                     / len(test_data_loader)
                
                results[dataset_path][model]['Accuracy'] += accuracy.item() \
                     / len(test_data_loader)

            logger(step)
    
    return results
