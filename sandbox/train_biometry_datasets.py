import time

import numpy as np

import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, segment_signal
from DeepLibphys.utils.functions.signal2model import Signal2Model

signal_dim = 64
hidden_dim = 256
mini_batch_size = 10
batch_size = 100
n_for_each = int(batch_size / 5)
window_size = 256
signal_directory = 'BIOMETRIC_ECGs_[{0}.{1}]'.format(batch_size, window_size)

signals_with_noise = [get_signals_tests(db.ecg_noisy_signals[noisy_index-1], signal_dim, type="ecg noise",
                                        noisy_index=noisy_index) for noisy_index in range(1,5)]
signals_without_noise = get_signals_tests(db.signal_tests, signal_dim, type="ecg")
signals = list(range(19))

for i in range(19):
    signals[i] = [signals_without_noise[0][i]] + [signals_with_noise[j][0][i] for j in range(4)]

# train each signal from fantasia
for i in range(2, 19):
    name = 'biometry_with_noise_' + str(i)
    signal2model = Signal2Model(name, signal_directory, mini_batch_size=mini_batch_size)
    model = GRU.LibphysMBGRU(signal2model)
    if i==2:
        model.load(dir_name=signal_directory, file_tag=model.get_file_tag(0,1000))

    model.train_block(signals[i], signal2model, n_for_each=n_for_each, loss_interval=1)
