import torch
from timeit import default_timer as timer
from src.evaluation.evaluation import evaluation
from src.training import training
from src.utils.utils import get_time
from src.evaluation.microphone_distance import eval_mic_distance




def main(
    train: bool = False, distance_eval: bool = False, evaluate: bool = False
) -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if distance_eval:
        eval_mic_distance(device)

    if train:
        training([
            'pgcc-phat-0',
            'pgcc-phat-1',
            'pgcc-phat-2',
            'ngcc-phat-0',
            'ngcc-phat-1',
            'ngcc-phat-2',
            'e2e-cnn',
        ], device)

    if evaluate:
        evaluation([
                'cc',
                'gcc_phat',
                'pgcc_phat',
                'ngcc_phat'
            ], [
                'e2e_cnn',
                'tde_ild',
                'srp_phat'
            ],
            device
        )


if __name__ == "__main__":
    start = timer()
    main()
    print(f'✨ Done in {get_time(timer() - start)}.')
