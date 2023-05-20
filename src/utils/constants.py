PLACEMENT = 'ceiling' # 'ceiling' or 'wall'

N_RECEIVERS = 4
R_RECEIVERS = 0.1

TRAINED_MODELS_PATH = 'trained-models'
TSP_DATASET_PATH = 'data/tsp-dataset'
ISL_DATASET_PATH = 'ISL-Dataset'
TESTING_SPEECH_DATASET_PATH = 'speech'


MIC_DISTANCE_FROM_WALL = 0.3
MIC_CENTROID_ELEVATION = 1.5

# Training parameters
TRAING_ROOM_DIMENSIONS = [3.3, 8, 8] if PLACEMENT == 'ceiling' else [9, 8, 3] 
TRAINING_N_OF_SIMULATIONS_PER_SAMPLE = 5
TRAINING_LABELS_PATH = f'data/training-labels-{PLACEMENT}.json'
TRAINING_SIGNALS_PATH = f'data/training-signal-{PLACEMENT}'
FINE_TUNE_LABELS_PATH = ''


# Mic distance evaluation parameters
DIST_EVAL_RECEIVERS_R = [0.1, 0.25, 0.5, 1]
DIST_EVAL_ROOM_DIMENSIONS = [(4, 10), (3, 8), (2, 4)]
DIST_EVAL_N_OF_SIMULATIONS_PER_SAMPLE = 3
DIST_EVAL_LABELS_PATH = 'data/dist-eval-labels.json'
DIST_EVAL_SIGNALS_PATH = 'data/dist-eval-signals'


# Synthetic evaluation parameters
TESTING_N_OF_SIMULATIONS_PER_SAMPLE = 5
TESTING_ROOM_DIMENSIONS = [3.3, 5, 7] if PLACEMENT == 'ceiling' else [7, 5, 3]
TESTING_PATH = f'eval-synthetic-{PLACEMENT}'

MIN_SPEAKER_HEIGHT = 1.2
MAX_SPEAKER_HEIGHT = 2

SOUND_VELOCITY = 343
SAMPLING_FREQUENCY = 48000
WINDOW_SIZE = 1 # In seconds
