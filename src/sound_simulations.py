import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily
import numpy as np
from typing import List, Optional, Tuple
from src.utils.types import Signal, Coordinates


def simulate_room(
    room_dimensions: Coordinates,
    source_position: Coordinates,
    receivers: List[Coordinates],
    signal: Tuple[Signal, int],
    rt60: Optional[float] = None,
    snr: Optional[int] = None,
    mic_directivity: bool = True,
    **kwargs,
) -> np.ndarray:
    materials: Optional[pra.Material] =\
      kwargs['materials'] if 'materials' in kwargs else None
    
    max_order: int = kwargs['max_order'] if 'max_order' in kwargs else 1
    rec_directivity = None

    if mic_directivity:
        rec_directivity = CardioidFamily(
            orientation=DirectionVector(
                # Directivity along the x-axis
                azimuth=0, colatitude=90, degrees=True
            ),
            pattern_enum=DirectivityPattern.CARDIOID,
        )

    if rt60:
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)
        materials = pra.Material(e_absorption)

    room = pra.ShoeBox(
        room_dimensions,
        air_absorption=True,
        fs=signal[1],
        sources=[pra.SoundSource(source_position, signal=signal[0])],
        materials=materials,
        max_order=max_order,
        # **kwargs,
    )
    room.add_microphone_array(np.transpose(receivers), rec_directivity)
    room.simulate(snr=snr)
    return room.mic_array.signals # type: ignore
