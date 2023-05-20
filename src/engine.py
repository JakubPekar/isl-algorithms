import torch
import torch.utils.data
from typing import Dict, Iterable, Optional
from collections import defaultdict
from src.configs.training_configs import training_datasets
from src.utils.ProgressBar import ProgressBar
from src.utils.WandbLogger import WandbLogger
from src.utils.types import Sample, TrainingParams
from src.utils.Checkpoint import Checkpoint
from src.utils.constants import PLACEMENT



def train_model(
    model: torch.nn.Module,
    name: str,
    config: TrainingParams,
    device: torch.device,
    fine: bool
) -> None:
    model.to(device)
    optimizer = config['optimizer']
    criterion = config['criterion']

    dataset = training_datasets(config['dataset_config'], fine)

    training_data, validation_data = torch.utils.data.random_split(
        dataset, [0.8, 0.2])

    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False,
    )

    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False,
    )

    logger = ProgressBar(len(train_data_loader), config['batch_size'])
    wandb_logger = WandbLogger(config['name'], config)
    checkpoint = Checkpoint(model, f"checkpoint/{name}-{PLACEMENT}.pt")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,  config['epochs'])

    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")

        train_one_epoch(
            model,
            train_data_loader,
            criterion,
            optimizer,
            device,
            logger,
            wandb_logger
        )

        score = evaluate(
            model,
            validation_data_loader,
            device,
            criterion,
            wandb_logger,
        )

        scheduler.step()

        checkpoint(score)
        if checkpoint.is_overfitting():
            break




def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable[Sample],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: Optional[ProgressBar] = None,
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    model.train(True)
    optimizer.zero_grad()

    for step, (samples, targets) in enumerate(data_loader, start=1):
        samples = samples.to(device, non_blocking=True).requires_grad_(True)
        targets = targets.to(device, non_blocking=True)

        output = model(samples)
        loss = criterion(output, targets)

        loss_value = loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if logger is not None:
            logger(step, loss=loss_value)
        if wandb_logger is not None:
            wandb_logger(loss=loss_value)



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader:torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    wandb_logger: Optional[WandbLogger] = None,
    **kwargs: torch.nn.Module,
) -> float:
    model.eval()
    loss = 0.0
    metrics: Dict[str, float] = defaultdict(float)

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(samples)
        loss += criterion(output, targets).item() / len(data_loader)

        for key, metric in kwargs.items():
            metrics[key] += metric(output, targets).item() / len(data_loader)

    metrics['val_loss'] = loss
    print(' - '.join(': '.join(map(str, x)) for x in metrics.items()))

    if wandb_logger is not None:
        wandb_logger(**metrics)

    return loss
