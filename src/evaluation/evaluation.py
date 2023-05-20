import torch
from typing import List
from src.evaluation.evaluate_tdoa_real import evaluate_tdoa_real
from src.evaluation.evaluate_tdoa_synthetic import evaluate_tdoa_synthetic
from src.Algorithms import Algorithms
from src.evaluation.evaluate_real import evaluate_real
from src.evaluation.evaluate_synthetic import evaluate_synthetic
from src.data_preparation import data_preparation
from src.utils.constants import PLACEMENT



def evaluation(
    tdoa_models: List[str],
    non_tdoa_models: List[str],
    device: torch.device
):
    data_preparation(isl_dataset=True, eval_synthetic_data=True)
    results = {}

    algorithms = Algorithms(device)

    print('Evaluating on synthetic data')
    results['synthetic'] = evaluate_synthetic(
        algorithms, tdoa_models + non_tdoa_models, device)
    
    results['synthetic-tdoa'] = evaluate_tdoa_synthetic(
        algorithms, tdoa_models, device)


    print('Evaluating on real data')
    real_datasets = [
        'ISL-Dataset/A320-labels.json',
        'ISL-Dataset/KD-labels.json',
        'ISL-Dataset/C525-labels.json',
        'ISL-Dataset/C511-labels.json',
    ]

    results['real'] = evaluate_real(
        algorithms, tdoa_models + non_tdoa_models, real_datasets, device)
    
    results['real-tdoa'] = evaluate_tdoa_real(
        algorithms, tdoa_models, real_datasets, device)


    with open(f'results/evaluation-{PLACEMENT}', "w") as file:
        file.write(str(results))

    print('Evaluation finished')
