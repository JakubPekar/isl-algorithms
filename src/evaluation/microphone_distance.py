import torch
import torch.utils.data
from src.losses.MOTPLoss import MOTPLoss
from src.losses.RMSELoss import RMSELoss
from src.data_preparation import data_preparation
from src.SyntheticDataset import SyntheticDataset
from src.models import GCC, PSO
from src.transforms import RandomCrop, ToTensor, Compose
from src.utils.constants import DIST_EVAL_LABELS_PATH, DIST_EVAL_RECEIVERS_R
from src.utils.utils import manual_seed, max_tau, true_tdoa
from src.utils.generators import circular_array_placement
from src.utils.ProgressBar import ProgressBar


ROOM_DIMENSIONS = [10, 8, 4]
BATCH_SIZE = 256


# Evaluation of mic distance using GCC-PHAT on 16 nad 48 kHz
def eval_mic_distance(device: torch.device):
    data_preparation(dist_eval_data=True)

    errors = {}
    criterion_rmse = RMSELoss()
    criteriom_mae = torch.nn.L1Loss()
    criterion_motp = MOTPLoss()
    

    for sf in [8000, 16000, 24000, 48000]:
        for i, r in enumerate(DIST_EVAL_RECEIVERS_R):
            manual_seed()

            model = GCC(
                device,
                max_tau=max_tau(r, sf),
                tdoa_estimate=True
            )

            mlat = PSO(device, sf)

            # The dimensions of the largest room in the dataset
            mlat.init(
                receivers=circular_array_placement(ROOM_DIMENSIONS, r), room_dimensions=ROOM_DIMENSIONS
            )

            dataset = SyntheticDataset(
                DIST_EVAL_LABELS_PATH,
                sf,
                lambda s, recs: torch.cat([
                    true_tdoa(s, recs, sf),
                    (s - torch.mean(recs.view(-1,3), dim=0)).view(1, 3) \
                        .repeat(6, 1)
                ], dim=-1),
                receiver_extractor=lambda x: torch.combinations(
                    torch.tensor(x[i * 4: (i + 1) * 4]),
                    2
                ),
                transform=Compose([
                    ToTensor(),
                    RandomCrop(2, sf),
                ]),
            )

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=False,
            )

            logger = ProgressBar(len(data_loader), BATCH_SIZE)

            loss_rmse = 0.0
            loss_mae = 0.0
            loss_motp = 0.0

            for  step, (samples, targets) in enumerate(data_loader, start=1):
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                gt_tdoas = targets[..., 0].unsqueeze(-1)
                gt_location = targets[..., 0, 1:]

                pred_tdoas = model(samples)

                loss_rmse += criterion_rmse(pred_tdoas, gt_tdoas).item() \
                    / len(data_loader)

                loss_mae += criteriom_mae(pred_tdoas, gt_tdoas).item() \
                    / len(data_loader)

                loss_motp += criterion_motp(
                    mlat(pred_tdoas), gt_location).item() / len(data_loader)
                
                logger(step, loss=loss_rmse)


            print(f"SF: {sf}, R: {r}, MSE: {loss_rmse}, MAE: {loss_mae}, MOTP: {loss_motp}")

            errors[f'{sf}_{r}_mse'] = loss_rmse
            errors[f'{sf}_{r}_mae'] = loss_mae
            errors[f'{sf}_{r}_motp'] = loss_motp


    with open('results/mic_distance_results', "w") as file:
        file.write(str(errors))
    
    return errors
