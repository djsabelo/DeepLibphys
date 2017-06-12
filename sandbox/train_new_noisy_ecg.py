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

signal_dim = 64
hidden_dim = 256
mini_batch_size = 16
batch_size = 256
window_size = 256
signal_directory = 'CLEAN_ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

# get noise from signals:
#
# signals_without_noise = []
# signals_noise = []
# SNR = []
# full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,21)))
# for file_path in full_paths[]:
#     # try:
#         print("Pre-processing signal - " + file_path)
#         signal = sio.loadmat(file_path)['val'][0][:1256]
#         # plt.plot(signal)
#         processed = process_dnn_signal(signal, signal_dim)
#         signals_without_noise.append(processed)
#         # plt.plot(processed)
#         # plt.show()
#         # plt.plot(signal[1000:5000])
#         # fig, ax = plt.subplots()
#         # # signalx = smooth(remove_moving_std(remove_moving_avg((signal - np.mean(signal)) / np.std(signal))))
#         # # signalx -= np.min(signalx)
#         # # signalx /= np.max(signalx)
#         # # signalx *= 64
#         #
#         # major_ticks = np.arange(0, 64)
#         # ax.set_yticks(major_ticks)
#         #
#         # plt.ylim([0, 15])
#         # plt.xlim([0, 140])
#         # ax.grid(True, which='both')
#         # plt.minorticks_on
#         # # ax.grid(which="minor", color='k')
#         # ax.set_ylabel('Class - k')
#         # ax.set_xlabel('Sample - n')
#         #
#         # plt.plot(signalx, label="Smoothed Signal", alpha=0.4)
#         # plt.plot(processed, label="Discretized Signal")
#         # # ticklines = ax.get_xticklines() + ax.get_yticklines()
#         # gridlines = ax.get_ygridlines()  # + ax.get_ygridlines()
#         # ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
#         #
#         # for line in gridlines:
#         #     line.set_color('k')
#         #     line.set_linestyle('-')
#         #     line.set_linewidth(1)
#         #     line.set_alpha(0.2)
#         #
#         # for label in ticklabels:
#         #     label.set_color('r')
#         #     label.set_fontsize('medium')
#         # plt.legend()
#         #
#         # plt.show()
#     # except:
#     #     print("Error")
#
# print("Saving signals...")
# np.savez("signals_without_noise.npz", signals_without_noise=signals_without_noise)
# #
print("Loading signals...")
noise_filename = "../data/ecg_noisy_signals.npz"
npzfile = np.load(noise_filename)
processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]

# plt.show()
for SNR, signals_with_noise in zip(SNRs[SNRs==10], processed_noise_array[SNRs==10]):
    for i, signal in zip(range(15,len(signals_with_noise)),signals_with_noise[15:]):
        name = 'ecg_SNR_' + str(SNR) + str(i+1)
        signal2model = Signal2Model(name, signal_directory, hidden_dim=hidden_dim, batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size)
        model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
        model.train(signal, signal2model)

