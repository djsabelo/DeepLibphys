import time

import numpy as np

import DeepLibphys
import DeepLibphys.utils.functions.database as db
from novainstrumentation import smooth
from DeepLibphys.utils.functions.common import *
import matplotlib.pyplot as plt
from DeepLibphys.utils.functions.signal2model import Signal2Model
import scipy.io as sio
import seaborn
import os
import DeepLibphys.models.LibphysMBGRU as GRU

def get_windows_left(array, signal, W, noverlap):
    windows = np.arange(0, len(signal) - W, int(W * noverlap))
    if array == []:
        return windows

    windows_left = []
    for w in windows:
        if w not in array:
            windows_left.append(w)

    return windows_left

def prepare_data(batch_size, signal_paths, history_of_indexes, sizes, n_per_signal=16, W=1024, noverlap = 0.33):
    minimum_matrix = []
    # decide signals to upload based on the minimum number of runs:
    for key, value in history_of_indexes.items():
        windows = np.arange(0, sizes[key] - W, int(W * noverlap))
        if len(windows) > len(value) or value == []:
            minimum_matrix.append(np.array([key, 0 if value == [] else len(value)]))
            # print(minimum_indexes)

    minimum_indexes = np.argsort(np.array(minimum_matrix)[:, 1])
    minimum_matrix = np.array(minimum_matrix)
    equal_indexes = np.where(minimum_matrix[:, 1] == minimum_matrix[minimum_indexes[0], 1])[0]
    if len(equal_indexes) > int(batch_size/n_per_signal):
        _random_indexes = np.random.permutation(equal_indexes)[:int(batch_size/n_per_signal)].tolist()
    else:
        indexes_to_consider = minimum_indexes[:int(batch_size/n_per_signal)]
        _random_indexes = np.random.permutation(indexes_to_consider).tolist()
    # print(np.array(_random_indexes))




    # if indexes_to_consider < (batch_size/n_per_signal):
    # print(_random_indexes)

    _random_indexes += np.random.permutation(minimum_indexes[int(batch_size/n_per_signal):]).tolist()

    # print(_random_indexes)
    # print(_random_indexes)
    last_index = int(batch_size / n_per_signal)

    random_indexes = _random_indexes[:last_index]
    # make sure there are enough indexes, if not, put other random signal
    validated = False
    while not validated:
        ok = True
        # load signals
        signals = [np.load("../data/processed/ALL/"+path)["signal"] for path in np.array(signal_paths)[random_indexes]]
        signals = np.array([signal if len(signal) > 1 else signal[0] for signal in signals])
        new_random_indexes = []
        for signal, index in zip(signals, random_indexes):
            windows_left = get_windows_left(history_of_indexes[index], signal, W, noverlap)
            # print(windows_left)
            if len(windows_left) > n_per_signal:
                new_random_indexes.append(index)
            else:
                ok = False
                new_random_indexes += [_random_indexes[last_index]]
                last_index += 1
        random_indexes = new_random_indexes
        validated = ok

    x_windows = np.zeros((batch_size, W))
    y_windows = np.zeros((batch_size, W))
    z = 0
    for signal, index in zip(signals, random_indexes):
        # windows = np.arange(0, len(signal)-W, int(W * noverlap))
        windows_left = get_windows_left(history_of_indexes[index], signal, W, noverlap)
        windows_left = np.random.permutation(windows_left)[:n_per_signal]
        # print(len(windows_left))
        x_windows[z * n_per_signal:z * n_per_signal + n_per_signal, :] = np.array(
            [np.array(signal[i:i + W]) for i in windows_left])
        y_windows[z * n_per_signal:z * n_per_signal + n_per_signal, :] = np.array(
            [np.array(signal[i + 1:i + W + 1]) for i in windows_left])

        # plt.plot(x_windows[z*n_per_signal, :])
        # plt.show()
        history_of_indexes[index] += windows_left.tolist()
        z += 1
    return x_windows, y_windows, random_indexes


def prepare_simple_data(batch_size, signal_paths, n_per_signal=16, W=1024, noverlap = 0.11):
    random_signals = np.random.permutation(signal_paths)[:int(batch_size/16)]
    signals = [np.load("../data/processed/ALL/" + path)["signal"] for path in np.array(random_signals)]
    signals = np.array([signal if len(signal) > 1 else signal[0] for signal in signals])

    x_windows = []
    y_windows = []
    for signal in signals:
        signal_rand_indexes = np.random.permutation(np.arange(0, len(signal) - W, int(noverlap*W)))[:n_per_signal]
        for ind in signal_rand_indexes:
            x_windows.append(signal[ind:ind + W])
            y_windows.append(signal[1 + ind:1 + ind + W])

    return np.array(x_windows), np.array(y_windows), random_signals


model_info = db.ecg_1024_256_RAW[0]
signal_dim = model_info.Sd
hidden_dim = model_info.Hd
mini_batch_size = 32
batch_size = 256
window_size = 1024
save_interval = 250
signal_directory = "ECG_CLUSTER[128.1024]"

signal_paths = []
for file in os.listdir("../data/processed/ALL/"):
    if file.endswith(".npz") and file != "history.npz":
        signal_paths.append(file)

signal2model = Signal2Model("generic_ecg", signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                            batch_size=batch_size,
                            mini_batch_size=mini_batch_size, window_size=window_size,
                            save_interval=save_interval, number_of_epochs=2000, lower_error=1e-9,
                            count_to_break_max=15, learning_rate_val=0.01)

model = GRU.LibphysMBGRU(signal2model)
limit = 3000
# model.start_time = time.time()
print(1)
i0 = 999
# history_of_indexes = {}
model.load(dir_name=signal_directory, file_tag=model.get_file_tag(i0, 0))
# history_of_indexes = np.load("../data/processed/ALL/history.npz")["history_of_indexes"]
#, history_of_indexes=history_of_indexes)
for i in range(i0, limit):
    returned = False
    while not returned:
        []
        x, y, ind = prepare_simple_data(batch_size, signal_paths, W=window_size)
        # print(np.shape(x))
        # print(np.shape(y))
        print("Signals Being Processed - {0} @ epoch {1}".format(ind, i))
        returned = True
        model.save(signal_directory, model.get_file_tag(i, 0))
        returned = model.train_model(x, y, signal2model, dataset=i)

    print(i)
    # np.savez("../data/processed/ALL/history.npz", history_of_indexes=history_of_indexes)



