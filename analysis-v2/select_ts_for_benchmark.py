import os
import json
import random
from pathlib import Path

SEED = 42
ALL_TS = '../data/timeseries/all'
DEST_TS = 'data/timeseries/benchmark'
PARTICIPANTS = ['michele', 'daniele', 'luca', 'vittorio']


def select_fork(ts_json, dest_dir):
    with open(ts_json) as f:
        # print(f)
        # print('asdf')
        ts = json.load(f)
    fork = random.choice(ts)
    with open(Path(dest_dir, ts_json.stem + '.json'), 'w') as f:
        # print(fork, f)
        json.dump([fork], f)


if __name__ == '__main__':
    random.seed(SEED)
    os.makedirs(DEST_TS, exist_ok=True)

    for p in PARTICIPANTS:
        os.makedirs(Path(DEST_TS, p), exist_ok=True)

    ts_jsons = list(Path(ALL_TS).glob('*.json'))
    # print(ts_jsons)
    random.shuffle(ts_jsons)

    for i, ts_json in enumerate(ts_jsons):
        dest_dir = Path(DEST_TS, PARTICIPANTS[i % len(PARTICIPANTS)])
        select_fork(ts_json, dest_dir)
