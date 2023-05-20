import json
from typing import Optional
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from pyannote.audio import Pipeline



LABELS_PATH = 'real-data/labels.json'
AUTH_TOKEN = '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'
SIGNALS_PATH = 'real-data/C525'
SF = 16000 # 16 kHz


"""
ACTOIN MANAGER
"""

MAIN_ACTIONS = {
    'k': 'keep',
    'd': 'discard',
    'e': 'edit',
    'p': 'play',
    's': 'show',
}

current_location: int = -1


def numeric_input(text: str, default: int = 0) -> int:
    return int(input(f'{text} ({default}): ') or default)

def get_action(actions: dict) -> str:
    action = ''
    while action not in actions:
        action = input(f"Select action: {', '.join([f'[{k}] - {actions[k]}' for k in actions])}: ")

    return action


def verify_action(text: str) -> bool:
    return input(f'{text} [y/n]: ') == 'y'


def get_label(start: int, end: int) -> Optional[tuple]:
    global current_location
    position = numeric_input('Position index', current_location)
    label = input('Label: ')
    current_location = position + 1
    return (start, end, position, label) if verify_action('Save changes?') else None



def action_manager(start: int, end: int, seq_id: int) -> Optional[tuple]:
    sd.play(signals[0][start:end], SF)
    action = get_action(MAIN_ACTIONS)
    response = None

    if action == 'k':
        response = get_label(start, end)
    elif action == 'd':
        if verify_action('Discard segment?'):
            return None
    elif action == 'e':
        start = numeric_input('Start', start)
        end = numeric_input('End', end)
        response = get_label(start, end)
    elif action == 'p':
        sd.play(signals[0][start:end], SF)
    elif action == 's':
        plot_segments(start, end, seq_id)

    return response if response is not None else action_manager(start, end, seq_id)




"""
MAIN
"""

pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization',
    use_auth_token=AUTH_TOKEN
)

signals = []

# Downsample the audio files to 16kHz
print('Preprocessing audio files...')
for i in range(1, 5):
    signal, sf = soundfile.read(f'{SIGNALS_PATH}_{i}.wav')
    signal = signal[::sf // SF]
    signals.append(signal)
    soundfile.write(f'{SIGNALS_PATH}_{i}_16.wav', signal, SF)
    print(f'Preprocessed {i}/4 files')


SIGNAL_LENGTH = len(signals[0])
ONE_SECOND = SF # type: ignore

# Process each file separately
print('Extracting voice segments...')
intersection = np.zeros(SIGNAL_LENGTH)
for i in range(1, 5):
    diarization = pipeline(f'{SIGNALS_PATH}_{i}_16.wav')

    mask = np.zeros(SIGNAL_LENGTH)
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = int(segment.start * SF) # type: ignore
        end = int(segment.end * SF) # type: ignore
        mask[start:end] = 1

    intersection += mask
    print(f'Extracted {i}/4 files')

mask = intersection > 1
indices = np.nonzero(mask[1:] != mask[:-1])[0] + 1

# Add first and last index if needed
if mask[0]:
    indices = np.insert(indices, 0, 0)
    if (len(indices) % 2) == 1:
       indices = np.append(indices, SIGNAL_LENGTH)

sequences = np.column_stack((indices[::2], indices[1::2]))

def plot_segments(start: int, end: int, seq_id: int) -> None:
    v_start = max(start - ONE_SECOND, 0)
    v_end = min(end + ONE_SECOND, SIGNAL_LENGTH)
    fig = plt.figure()
    main_ax = fig.add_subplot(311)
    main_ax.plot(signals[0])
    for prev_s, prev_e in sequences[:seq_id + 1]:
        main_ax.axvspan(prev_s, prev_e, facecolor='#d62728', alpha=0.4)
    for i in range(0, 4):
        ax = fig.add_subplot(3, 2, i + 3)
        ax.axvline(start - v_start, color='r')
        ax.axvline(end - v_start, color='r')
        ax.plot(signals[i][v_start:v_end])
    plt.show()
    plt.close(fig)


labels = np.array([], dtype='<U69')

print('Select propriet segments')
for seq_id, (start, end) in enumerate(sequences):
    print(f'{start} - {end}')
    # plot_segments(start, end, seq_id)
    response = action_manager(start, end, seq_id)
    print(response)
    if response is not None:
        labels = np.append(labels, response)        


transorm_labels = np.vectorize(lambda x: {'start': int(x[0]), 'end': int(x[1]), 'pos': int(x[2]), 'label': x[3]}, signature='(n)->()')

labels = transorm_labels(np.split(labels, len(labels) // 4))

with open(LABELS_PATH, "w") as file:
        file.write(json.dumps(labels.tolist()))
