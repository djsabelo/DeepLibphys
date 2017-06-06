import time

import numpy as np

import DeepLibphys
import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from novainstrumentation import smooth
from DeepLibphys.utils.functions.common import *
import matplotlib.pyplot as plt
from DeepLibphys.utils.functions.signal2model import Signal2Model
import scipy.io as sio
import seaborn


def make_noise_signals(full_paths, max_target_SNR=16, signal_dim=64):
    target_SNR_array = np.arange(max_target_SNR, max_target_SNR-10, -1)
    N_SIGNALS, N_NOISE, N_SAMPLES = len(full_paths), len(target_SNR_array), 0
    processed_noise_array = np.zeros((N_NOISE, N_SIGNALS, len(sio.loadmat(full_paths[0])['val'][0])))
    SNR = np.zeros(N_NOISE)
    for i, file_path in zip(range(len(full_paths)), full_paths):
        print("Processing " + file_path)
        signal = sio.loadmat(file_path)['val'][0]
        signal = remove_moving_std(remove_moving_avg((signal - np.mean(signal)) / np.std(signal)))
        smoothed_signal = smooth(signal)

        SNR = int(calculate_signal_to_noise_ratio(signal, smoothed_signal))

        last_std = 0.0001
        j = 0
        for target_SNR in target_SNR_array:
            signal_with_noise, last_std = make_noise_vectors(signal, smoothed_signal, target_SNR, last_std=last_std)
            processed_noise_array[j, i, :] = process_dnn_signal(signal_with_noise, signal_dim)
            j += 1

    return processed_noise_array, target_SNR_array


def calculate_signal_to_noise_ratio(signal, smoothed_signal):
    signal = signal[13000:19000]
    smoothed_signal = smoothed_signal[13000:19000]
    noise_signal = smoothed_signal - signal
    SMOOTH_AMPLITUDE = np.max(smoothed_signal) - np.min(smoothed_signal)
    NOISE_AMPLITUDE = np.mean(np.abs(noise_signal)) * 2
    SNR = (10 * np.log10(SMOOTH_AMPLITUDE / NOISE_AMPLITUDE))
    return SNR


def make_noise_vectors(signal, smoothed_signal, target_SNR, last_std=0.0001):
    SNR = int(calculate_signal_to_noise_ratio(signal, smoothed_signal))
    added_noise = np.random.normal(0, last_std, len(signal))
    signal_with_noise = np.array(np.array(signal) + added_noise)
    print("Processing SNR {0}".format(target_SNR))
    if SNR < target_SNR:
        signal = smoothed_signal

    while int(SNR*100) != target_SNR*100:

        added_noise = np.random.normal(0, last_std, len(signal))
        signal_with_noise = np.array(np.array(signal) + added_noise)
        SNR = calculate_signal_to_noise_ratio(signal_with_noise, smoothed_signal)

        if int(SNR*100) > target_SNR*100:
            last_std *= 2
        else:
            last_std *= 0.2

    return signal_with_noise, last_std

if __name__ == "__main__":
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 16
    batch_size = 128
    window_size = 1024
    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size, signal_dim)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    noise_filename = dir_name + "/signals_without_noise_[{}].npz".format(signal_dim)

    # get noisy signals:
    full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1, 21)))
    # processed_noise_array, SNRs = make_noise_signals(full_paths, max_target_SNR=12, signal_dim=signal_dim)
    # print("Saving signals...")
    #
    # np.savez(noise_filename,
    #          processed_noise_array=processed_noise_array, SNRs = SNRs)

    print("Loading signals...")
    # noise_filename = "../data/ecg_noisy_signals.npz"
    npzfile = np.load(noise_filename)
    processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]

    # plt.show()
    for SNR, signals_with_noise in zip(SNRs, processed_noise_array):
        for i, signal in zip(range(len(signals_with_noise)), signals_with_noise):
            if SNR == 12 and i < 7 :
                pass
            elif SNR < 8:
                name = 'ecg_' + str(i+1) + '_SNR_' + str(SNR)
                signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                            batch_size=batch_size, mini_batch_size=mini_batch_size, window_size=window_size)
                running_ok = False
                while not running_ok:
                    model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
                    running_ok = model.train(signal, signal2model)

