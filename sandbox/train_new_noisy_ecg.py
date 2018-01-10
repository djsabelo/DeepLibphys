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

def load_noisy_fantasia_signals(SNRx=None):
    clean_list = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']
    clean = np.zeros((1, len(clean_list), len(clean_list[0])))
    for i, c in enumerate(clean_list):
            clean[0, i, :] = np.array(c[-len(clean_list[0]):], dtype=np.int)

    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(128, 1024)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory
    noise1 = np.load(dir_name + "/signals_without_noise_[{0}].npz".format(256))['processed_noise_array']
    noise2 = np.load(dir_name + "/signals_with_noise_2_[{0}].npz".format(256))['processed_noise_array']
    SNRs = np.load(dir_name + "/signals_without_noise_[{0}].npz".format(256))['SNRs']

    if SNRx is not None:
        noise1 = noise1[np.where(np.logical_and(SNRs >= SNRx[0], SNRs <= SNRx[1]))[0]]
        noise2 = noise2[np.where(np.logical_and(SNRs >= SNRx[0], SNRs <= SNRx[1]))[0]]

    return np.vstack((clean, np.hstack((noise1, noise2))))

def make_noise_signals(full_paths, max_target_SNR=16, signal_dim=64):
    target_SNR_array = np.arange(max_target_SNR, max_target_SNR-10, -1)
    N_SIGNALS, N_NOISE, N_SAMPLES = len(full_paths), len(target_SNR_array), 0
    processed_noise_array = np.zeros((N_NOISE, N_SIGNALS, len(sio.loadmat(full_paths[0])['val'][0])))
    SNR = np.zeros(N_NOISE)
    SNRs = np.zeros(N_SIGNALS)
    for i, file_path in zip(range(len(full_paths)), full_paths):
        print("Processing " + file_path)
        signal = sio.loadmat(file_path)['val'][0]
        signal = remove_moving_std(remove_moving_avg((signal - np.mean(signal)) / np.std(signal)))
        smoothed_signal = smooth(signal)

        SNR = int(calculate_signal_to_noise_ratio(signal, smoothed_signal))
        SNRs[i] = SNR
        # last_std = 0.0001
        # j = 0
        # for target_SNR in target_SNR_array:
        #     signal_with_noise, last_std = make_noise_vectors(signal, smoothed_signal, target_SNR, last_std=last_std)
        #     processed_noise_array[j, i, :] = process_dnn_signal(signal_with_noise, signal_dim, confidence=0.005)[:np.shape(processed_noise_array)[2]]
        #     j += 1
    print(np.mean(SNRs))
    print(np.std(SNRs))
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

    # noise_filename = dir_name + "/signals_with_noise_2_[{0}].npz".format(signal_dim)
    # noise_filename = dir_name + "/signals_without_noise_[{0}].npz".format(signal_dim)
    # get noisy signals:
    # full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1, 41)))
    # processed_noise_array, SNRs = make_noise_signals(full_paths, max_target_SNR=12, signal_dim=signal_dim)
    # print("Saving signals...")
    #
    # np.savez(noise_filename,
    #          processed_noise_array=processed_noise_array, SNRs = SNRs)

    print("Loading signals...")
    # noise_filename = "../data/ecg_noisy_signals.npz"
    processed_noise_array = load_noisy_fantasia_signals(SNRx=[6, 9])
    # processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]

    # snr1, snr2 = 7, 5
    # SNRs = [8]#, SNRs[5]]
    signal2model = Signal2Model("", signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                batch_size=batch_size, mini_batch_size=mini_batch_size, window_size=window_size)
    model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)

    SNRs = ["RAW"] + [str(i) for i in range(9, 6, -1)]
    for SNR, signals_with_noise in zip(SNRs, processed_noise_array):
        for i, signal in zip(range(len(signals_with_noise)), signals_with_noise):
            running_ok = False
            if SNR is not "RAW":
                u = i+1
                # if np.logical_and(SNR == 9, u > 4) and np.logical_and(SNR == 9, u < 21):
                name = 'ecg_' + str(u) + '_SNR_' + str(SNR)
                print(name)
                signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                            batch_size=batch_size, mini_batch_size=mini_batch_size, window_size=window_size)
                model.model_name = name
                while not running_ok:
                    # model.model_name = 'ecg_' + str(u) + '_SNR_' + str(12)
                    try:
                        model.load(model.get_file_tag(-5, -5), signal_directory)
                        running_ok = True
                    except:
                        model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
                        running_ok = model.train(signal, signal2model)

