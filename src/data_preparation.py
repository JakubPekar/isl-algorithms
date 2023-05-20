import os
import json
import soundfile
import numpy as np
from typing import Callable
from itertools import cycle
from src.configs.room_simulation import evaluation_room_simulation_config, training_room_simulation_config, multi_dist_setup_room_simulation_config
from src.utils.constants import DIST_EVAL_LABELS_PATH, DIST_EVAL_N_OF_SIMULATIONS_PER_SAMPLE, DIST_EVAL_SIGNALS_PATH, ISL_DATASET_PATH, TESTING_N_OF_SIMULATIONS_PER_SAMPLE, TESTING_PATH, TESTING_ROOM_DIMENSIONS, TESTING_SPEECH_DATASET_PATH, TRAINING_N_OF_SIMULATIONS_PER_SAMPLE, TRAINING_LABELS_PATH, TRAINING_SIGNALS_PATH, TSP_DATASET_PATH
from src.sound_simulations import simulate_room
from src.utils.ProgressBar import ProgressBar
from src.utils.types import RoomSimulationConfig
from src.utils.generators import generate_random_position
from src.utils.utils import manual_seed



# Flac supports up to 8 channels only
def modify_file_name(file_name: str, i: int, channles: int) -> str:
    ending = 'wav' if channles > 8 else 'flac'
    return file_name.split('.')[0] + '_' + str(i) + f'.{ending}'




# Remaps nested dataset into a flat structure
def flat_map_data(path: str, destination: str) -> None:
    for f in os.listdir(path):
        f_path = os.path.join(path, f)

        if os.path.isdir(f_path):
            flat_map_data(f_path, destination)

        # Copy only .wav and .flac files
        elif f.split('.')[-1] in ['flac', 'wav']:
            os.popen(f'cp {f_path} {os.path.join(destination, f)}')




def data_simulation(
    room_sim_config: Callable[[str], RoomSimulationConfig],
    labels_path: str,
    signals_path: str,
    dataset_path: str,
    n_of_simulations: int
):
    if os.path.exists(labels_path):
        return
    
    if not os.path.exists(dataset_path):
        raise Exception('Dataset not found.')
    
    print(f'Preparing synthetic data from {dataset_path} using room simulation...')

    os.system(f'mkdir {signals_path}')
    logger = ProgressBar(len(os.listdir(dataset_path)), n_of_simulations)
    labels = []

    for step, file in enumerate(os.listdir(dataset_path), start=1):
        for i in range(n_of_simulations):
            room_config = room_sim_config(os.path.join(dataset_path, file))
            signal = simulate_room(**room_config)
            new_path = os.path.join(
                signals_path,
                modify_file_name(file, i, len(room_config['receivers']))
            )

            try:
                # Save signal
                soundfile.write(
                    new_path, np.transpose(signal), room_config['signal'][1]
                )

                # Save room config, replace signal with path to new signal
                room_config['signal'] = new_path # type: ignore
                labels.append(room_config)

            except Exception as e:
                print('An error occured:', e)
                

        # Log progress
        logger(step)

        # Save labels every 50 steps
        if (step + 1) % 50:
            with open(labels_path, "w") as file:
                file.write(json.dumps(labels))
            # Sometimes the file is not saved correctly
            with open('data/tempcopy', "w") as file:
                file.write(json.dumps(labels)) 
            

    # Save labels
    with open(labels_path, "w") as file:
        file.write(json.dumps(labels))
    
    # Remove temp file
    os.system('rm data/tempcopy')

    print('Data simulation finished.')




def download_tsp_dataset() -> None:
    if os.path.exists(TSP_DATASET_PATH):
        return
    
    print('Downloading TSP Speech Dataset in 48 kHz...')

    # Download TSP Speech Dataset in 48 kHz 
    os.system(f'mkdir {TSP_DATASET_PATH}')
    os.system('wget https://www.mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/48k.zip')
    os.system('unzip 48k.zip -d data/temp')
    os.system('rm 48k.zip')

    print('TSP dataset downloaded. Extracting TSP dataset...')

    # Flatten nested dataset
    flat_map_data('data/temp', TSP_DATASET_PATH)
    os.system('rm -Rf data/temp')
    print('TSP datset extracted.')



def download_isl_dataset() -> None:
    if os.path.exists(ISL_DATASET_PATH):
        return

    print('Downloading ISL-Dataset')
    # Download git ISL-Dataset 
    os.system('git clone https://github.com/JakubPekar/ISL-Dataset.git')
    print('ISL-Dataset downloaded.')



def prepare_synthetic_eval_data() -> None:
    if os.path.exists(TESTING_PATH):
        return

    # Reproducibility
    manual_seed()

    # Evaluate the influence of SNR and RT60 with fixed source position
    source_pos_iterator = cycle([
        generate_random_position(TESTING_ROOM_DIMENSIONS) \
            for _ in TESTING_N_OF_SIMULATIONS_PER_SAMPLE * os.listdir(TESTING_SPEECH_DATASET_PATH)
    ])

    print('Preparing synthetic evaluation data...')
    os.system(f'mkdir {TESTING_PATH}')

    print('Prepairing an anechoic room with variouse SNR levels...')
    for snr in np.linspace(-10, 30, 9):
        room_config: Callable[[str], RoomSimulationConfig] \
                = lambda signal_path: {
                    **evaluation_room_simulation_config(signal_path),
                    'source_position': next(source_pos_iterator),
                    'rt60': None,
                    'snr': snr,
                    'max_order': 0, 
                } # type: ignore
        
        data_simulation(
            room_config,
            os.path.join(TESTING_PATH, f"SNR_{snr}.json"),
            os.path.join(TESTING_PATH, f"SNR_{snr}"),
            TESTING_SPEECH_DATASET_PATH,
            TESTING_N_OF_SIMULATIONS_PER_SAMPLE,
        )

    print('Prepairing a reverberant room with variouse RT60...')
    for rt60 in np.linspace(0.2, 1, 9):
        room_config: Callable[[str], RoomSimulationConfig] \
                = lambda signal_path: {
                    **evaluation_room_simulation_config(signal_path),
                    'source_position': next(source_pos_iterator),
                    'rt60': rt60,
                    'snr': 30, # 30 dB
                } # type: ignore
        
        data_simulation(
            room_config,
            os.path.join(TESTING_PATH, f"RT60_{rt60}.json"),
            os.path.join(TESTING_PATH, f"RT60_{rt60}"),
            TESTING_SPEECH_DATASET_PATH,
            TESTING_N_OF_SIMULATIONS_PER_SAMPLE,
        )
    
    print('Synthetic evaluation data prepared.')




def prepare_synthetic_training_data() -> None:
    data_simulation(
        training_room_simulation_config,
        TRAINING_LABELS_PATH,
        TRAINING_SIGNALS_PATH,
        TSP_DATASET_PATH,
        TRAINING_N_OF_SIMULATIONS_PER_SAMPLE
    )



def prepare_synthetic_dist_eval_data() -> None:
    data_simulation(
        multi_dist_setup_room_simulation_config,
        DIST_EVAL_LABELS_PATH,
        DIST_EVAL_SIGNALS_PATH,
        TSP_DATASET_PATH,
        DIST_EVAL_N_OF_SIMULATIONS_PER_SAMPLE
    )







def data_preparation(
    tsp_dataset: bool = False,
    isl_dataset: bool = False,
    training_data: bool = False,
    dist_eval_data: bool = False,
    eval_synthetic_data: bool = False,
) -> None:
    print('Preparing datasets...')

    if tsp_dataset:
        download_tsp_dataset()

    if isl_dataset:
        download_isl_dataset()

    if training_data:
        prepare_synthetic_training_data()
        download_isl_dataset()

    if dist_eval_data:
        prepare_synthetic_dist_eval_data()

    if eval_synthetic_data:
        prepare_synthetic_eval_data()

    print('Datasets are ready.')


# Expose only data_preparation function
__all__ = [
    'data_preparation'
]