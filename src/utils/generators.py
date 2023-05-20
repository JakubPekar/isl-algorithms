import torch
import random
from typing import List, Tuple
from src.utils.constants import MAX_SPEAKER_HEIGHT, MIC_CENTROID_ELEVATION, MIC_DISTANCE_FROM_WALL, MIN_SPEAKER_HEIGHT, N_RECEIVERS, PLACEMENT
from src.utils.types import Coordinates
from src.utils.utils import circular_array



def circular_array_placement(
    room_dimensions: Coordinates, radius: float
) -> torch.Tensor:
    centroid = torch.tensor([
        MIC_DISTANCE_FROM_WALL,
        room_dimensions[1] / 2,
        MIC_CENTROID_ELEVATION
    ])

    return centroid + circular_array(N_RECEIVERS, radius)




def generate_random_position(
    room_dimensions: Coordinates,
    grid_density: float = 0.1
) -> Coordinates:
    samples_per_unit = 1 / grid_density

    r_pos = lambda l, u: random.randint(
        int(l* samples_per_unit),
        int(u * samples_per_unit)
    ) / samples_per_unit

    if PLACEMENT == 'ceiling':
        return [
            r_pos(
                room_dimensions[0] -  MAX_SPEAKER_HEIGHT,
                room_dimensions[0] - MIN_SPEAKER_HEIGHT
            ),
            r_pos(0, room_dimensions[1]),
            r_pos(0, room_dimensions[2])
        ]

    return [
        r_pos(MIC_DISTANCE_FROM_WALL, room_dimensions[0]),
        r_pos(0, room_dimensions[1]),
        r_pos(MIN_SPEAKER_HEIGHT, MAX_SPEAKER_HEIGHT)
    ]



def generate_room_dimensions(
    room_dimensions: List[Tuple[int, int]]
) -> Coordinates:
    return list(map(lambda x: random.randint(*x), room_dimensions))
