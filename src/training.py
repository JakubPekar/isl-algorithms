import os
import torch
from typing import List
from src.engine import train_model
from src.configs.training_configs import get_training_configs
from src.data_preparation import data_preparation
from src.utils.constants import PLACEMENT, TRAINED_MODELS_PATH



def save_checkpoint(model: torch.nn.Module, name: str, fine: bool) -> None:
    model.load_state_dict(torch.load(f"checkpoint/{name}-{PLACEMENT}.pt"))

    torch.save(model, os.path.join(
        TRAINED_MODELS_PATH, f"{name}_{PLACEMENT}{'_fine' if fine else ''}"))



def training(models: List[str], device: torch.device):
    data_preparation(tsp_dataset=True, training_data=True)
    training_configs = get_training_configs()

    for model_name in models:
        model, config = training_configs[model_name](device)
        
        print('Training model: ' + model_name)
        train_model(model, model_name, config.copy(), device, fine=False)
        save_checkpoint(model, model_name, fine=False)

        print(f'Training of {model_name} finished')
