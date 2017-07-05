import time

import numpy as np

from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import Signal2Model

import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db

signal_dim = 64
hidden_dim = 256
mini_batch_size = 10
batch_size = 128
n_for_each = int(batch_size / 5)
window_size = 256
signal_directory = 'BIOMETRIC_ECGs_[{0}.{1}]'.format(batch_size, window_size)

# signals_with_noise = [get_signals_tests(db.ecg_noisy_signals[noisy_index-1], signal_dim, type="ecg noise",
#                                         noisy_index=noisy_index) for noisy_index in range(1,5)]
# signals_without_noise = get_signals_tests(db.signal_tests, signal_dim, type="ecg")
# signals = list(range(19))
#
# for i in range(19):
#     signals[i] = [signals_without_noise[0][i]] + [signals_with_noise[j][0][i] for j in range(4)]
#
# # train each signal from fantasia
# for i in range(2, 19):
#     name = 'biometry_with_noise_' + str(i)
#     signal2model = Signal2Model(name, signal_directory, mini_batch_size=mini_batch_size)
#     model = GRU.LibphysMBGRU(signal2model)
#     if i==2:
#         model.load(dir_name=signal_directory, file_tag=model.get_file_tag(0,1000))
#
#     model.train_block(signals[i], signal2model, n_for_each=n_for_each, loss_interval=1)

signal_dim = 64
hidden_dim = 256
mini_batch_size = 15
batch_size = 150
n_for_each = 150/5
signal_directory = 'NOISE_ECGs_[{0}.{1}]'.format(batch_size, window_size)
noise_filename = "../data/ecg_noisy_signals.npz"
npzfile = np.load(noise_filename)
processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]

ecgs = np.load("signals_without_noise.npz")['signals_without_noise']
z = 0

for i, ecg in zip(range(z, len(ecgs)), ecgs[z:]):
    name = 'noisy_ecg_' + str(i+1)
    signals= [ecg] + [pna[i] for pna in processed_noise_array]
    signal2model = Signal2Model(name, signal_directory, mini_batch_size=mini_batch_size)
    model = GRU.LibphysMBGRU(signal2model)
    model.train_block(signals, signal2model, n_for_each=n_for_each)
        # train(ecg, signal2model)