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


def process_and_save_fantasia(plot=False, signal_dim=64):
    """
    insert noise into fantasia dataset
    
    :param plot: 
    :param signal_dim: 
    :return: 
    """

    signals_without_noise = []
    signals_noise = []
    SNR = []
    full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,21)))
    for file_path in full_paths:
        # try:
        print("Pre-processing signal - " + file_path)
        signal = sio.loadmat(file_path)['val'][0][:1256]
        # plt.plot(signal)
        processed = process_dnn_signal(signal, signal_dim)
        signals_without_noise.append(processed)
        if plot:
            plt.plot(processed)
            plt.show()
            plt.plot(signal[1000:5000])
            fig, ax = plt.subplots()
            major_ticks = np.arange(0, 64)
            ax.set_yticks(major_ticks)
            plt.ylim([0, 15])
            plt.xlim([0, 140])
            ax.grid(True, which='both')
            plt.minorticks_on
            # ax.grid(which="minor", color='k')
            ax.set_ylabel('Class - k')
            ax.set_xlabel('Sample - n')
            plt.plot(signalx, label="Smoothed Signal", alpha=0.4)
            plt.plot(processed, label="Discretized Signal")
            # ticklines = ax.get_xticklines() + ax.get_yticklines()
            gridlines = ax.get_ygridlines()  # + ax.get_ygridlines()
            ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

            for line in gridlines:
                line.set_color('k')
                line.set_linestyle('-')
                line.set_linewidth(1)
                line.set_alpha(0.2)

            for label in ticklabels:
                label.set_color('r')
                label.set_fontsize('medium')
            plt.legend()

            plt.show()

        signalx = smooth(remove_moving_std(remove_moving_avg((signal - np.mean(signal)) / np.std(signal))))

    print("Saving signals...")
    np.savez("signals_without_noise.npz", signals_without_noise=signals_without_noise)


def train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, signals,save_interval,signal_dim):
    models = db.resp_64_models
    for i, signal, model_info in zip(indexes, signals, models):
        name = 'resp_' + str(i)
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim, batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval)
        print("Compiling Model {0}".format(name))
        model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
        print("Initiating training... ")
        model.load(dir_name=model_info.directory)
        model.train(signal, signal2model)

signal_dim = 64
hidden_dim = 256
mini_batch_size = 16
batch_size = 128
window_size = 1024
save_interval = 1000
signal_directory = 'SYNTHESIS[{0}.{1}]'.format(batch_size, window_size)

indexes = np.arange(1, 21, dtype=np.int)

print("Loading signals...")
# signals = np.load("../data/signals_without_noise.npz")['signals_without_noise'][indexes]
# x_train, y_train = get_fantasia_dataset(signal_dim, indexes, db.fantasia_resp[0].directory, peak_into_data=False)
# np.savez("../data/Fantasia_RESP_[64].npz", x_train=x_train, y_train=y_train)
# x_train, y_train = np.load("../data/Fantasia_RESP_[64].npz")['x_train'][indexes-1], \
#                    np.load("../data/Fantasia_RESP_[64].npz")['y_train'][indexes-1]
#
# # i=0
# # for signal in x_train:
# #     i += 1
# #     plt.figure(i)
# #     plt.plot(signal[1000:1000 + 3000])
# # plt.show()
# train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, x_train, save_interval,
#                signal_dim)

indexes = np.arange(1, 21, dtype=np.int)

print("Loading signals...")
# signals = np.load("../data/signals_without_noise.npz")['signals_without_noise'][indexes]
# x_train, y_train = get_fantasia_dataset(signal_dim, indexes, db.fantasia_resp[0].directory, peak_into_data=False)
# np.savez("../data/Fantasia_RESP_[64].npz", x_train=x_train, y_train=y_train)
x_train, y_train = np.load("../data/Fantasia_RESP_[64].npz")['x_train'][indexes-1], \
                   np.load("../data/Fantasia_RESP_[64].npz")['y_train'][indexes-1]

# i=0
# for signal in x_train:
#     i += 1
#     plt.figure(i)
#     plt.plot(signal[1000:1000 + 3000])
# plt.show()
train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, x_train, save_interval,
               signal_dim)