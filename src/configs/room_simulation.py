import random
import soundfile
import numpy as np
from src.utils.constants import DIST_EVAL_RECEIVERS_R, DIST_EVAL_ROOM_DIMENSIONS, TESTING_ROOM_DIMENSIONS, R_RECEIVERS, TRAING_ROOM_DIMENSIONS
from src.utils.generators import circular_array_placement, generate_random_position, generate_room_dimensions
from src.utils.types import PartialRoomSimulationConfig, RoomSimulationConfig



def training_room_simulation_config(signal_path: str) -> RoomSimulationConfig:
    return {
        'room_dimensions': TRAING_ROOM_DIMENSIONS,
        'source_position': generate_random_position(TRAING_ROOM_DIMENSIONS),
        'receivers': circular_array_placement(
                TRAING_ROOM_DIMENSIONS, R_RECEIVERS
            ).tolist(),
        'signal': soundfile.read(signal_path),
        'rt60': random.randint(2, 8) / 10,
        'snr': random.randint(0, 30),
    }



# Simulate room with distances of [0.1, 0.25, 0.5, 1] m from the mic centroid
def multi_dist_setup_room_simulation_config(
    signal_path: str
) -> RoomSimulationConfig:
    room_dimensions = generate_room_dimensions(DIST_EVAL_ROOM_DIMENSIONS)
    
    return {
        'room_dimensions': room_dimensions,
        'source_position': generate_random_position(room_dimensions),
        'receivers': np.concatenate([
            circular_array_placement(
                room_dimensions, r
            ) for r in DIST_EVAL_RECEIVERS_R
        ]).tolist(),
        'signal': soundfile.read(signal_path),
        'rt60': random.randint(2, 8) / 10,
        'snr': random.randint(0, 30),
    }



def evaluation_room_simulation_config(
    signal_path: str
) -> PartialRoomSimulationConfig:
    return {
        'room_dimensions': TESTING_ROOM_DIMENSIONS,
        'receivers': circular_array_placement(
            TESTING_ROOM_DIMENSIONS, R_RECEIVERS
        ).tolist(),
        'signal': soundfile.read(signal_path),
    }