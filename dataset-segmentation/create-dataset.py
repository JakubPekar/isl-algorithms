from typing import List
import soundfile
import json
import os
import numpy as np


LABELS_PATH = 'real-data/labels.json'
SF = 192000
SF_FACTOR = 12
MIC_DISTANCE = 0.1

SIGNALS_PATH = 'real-data/C525'
OUTPUT_PATH = 'ISL-Dataset/C525'
ROOM_DIMENSIONS = [9, 4.90, 3.25]
MIC_CENTROID = [0.38, 2.37, 1.5]

POSITION_MAP = {
        1: [3.32, 0.29],
        2: [3.21, 1.57],
        3: [3.18, 2.85],
        4: [4.46, 0.90],
        5: [4.39, 2.14],
        6: [4.46, 3.42],
        7: [5.60, 0.50],
        8: [5.67, 1.72],
        9: [5.71, 3.03],
        10: [6.98, 1.03],
        11: [7.01, 2.37],
        12: [6.99, 3.63],
        13: [8.49, 1.77],
        14: [8.44, 3.01],
    }

LABELS_MAP = []

def get_position(key: int, index: int) -> List[float]:
    x, y = POSITION_MAP[key]
    x, y = x - 0.37, y - 2.7
    z = -0.37 if 2 * key < index - 5 else 0.12
    return np.round([x, y, z], decimals=2).tolist()


signals = np.array([])

print('Loading files...')
for i in range(1, 5):
    signal, sf = soundfile.read(f'{SIGNALS_PATH}_{i}.wav')
    signals  = np.hstack((signals, signal))
    print(f'Loaded {i}/4 files')


signals = signals.reshape((4, -1)).T

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)


def mic_array():
    x, y, z = MIC_CENTROID
    return [
        [x, y - MIC_DISTANCE, z],
        [x, y, z + MIC_DISTANCE],
        [x, y + MIC_DISTANCE, z],
        [x, y, z - MIC_DISTANCE],
    ]


print('Creating dataset...')
os.makedirs(OUTPUT_PATH, exist_ok=True)

data = {
    'room-dimensions': ROOM_DIMENSIONS,
    'receivers': mic_array(),
    'data': []
}

for i, label in enumerate(labels):
    f_name = f'{OUTPUT_PATH}/{i}.flac'

    soundfile.write(
        f_name,
        signals[label['start'] * SF_FACTOR:label['end'] * SF_FACTOR],
        SF
    )
    
    data['data'].append({
        'source-position': get_position(label['pos'], i),
        'label': label['label'] if label['label'] else LABELS_MAP[label['pos'] - 1],
        'signal': f_name,
    })

    print(f'{i + 1}/{len(labels)}')
    

with open(f'{OUTPUT_PATH}-labels.json', "w") as file:
    file.write(json.dumps(data))
