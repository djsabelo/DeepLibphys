import time

import numpy as np

import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from novainstrumentation import smooth
from DeepLibphys.utils.functions.common import get_signals_tests, segment_signal, get_fantasia_full_paths
import matplotlib.pyplot as plt
from DeepLibphys.utils.functions.signal2model import Signal2Model
import scipy.io as sio
import seaborn

signal_dim = 64
hidden_dim = 256
mini_batch_size = 10
batch_size = 100
n_for_each = int(batch_size / 5)
window_size = 256
signal_directory = 'NOISE_BIOMETRIC_ECGs_[{0}.{1}]'.format(batch_size, window_size)

# get noise from signals:

signals_without_noise = []
signals_noise = []
SNR = []
full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,20)))
for file_path in full_paths:
    signal = sio.loadmat(file_path)['val'][0]
    signal = (signal - np.mean(signal))/np.std(signal)
    signals_noise.append(np.array(signal) + np.random.normal(0, 0.8, len(signal)))
    signal = signals_noise[-1]
    signals_without_noise.append(signal)
    smoothed_signal = smooth(signal)
    noise = smoothed_signal - signals_without_noise[-1]
    print(np.std(signal))
    SMOOTH_AMPLITUDE = np.max(smoothed_signal) - np.min(smoothed_signal)
    NOISE_AMPLITUDE = np.mean(np.abs(noise))*2

    SNR.append(10 * np.log10(SMOOTH_AMPLITUDE/NOISE_AMPLITUDE))

    # print(SNR[-1]*10, "-" + str(SMOOTH_AMPLITUDE/NOISE_AMPLITUDE))
    # print(np.mean(SNR * 10))
    plt.plot(noise[:1000],'r',smoothed_signal[:1000],'k')#, signals_noise[-1][:1000], 'b')
    plt.show()

print(np.mean(np.array(SNR)))
    # plt.show()

signals_without_noise = np.array(signals_without_noise)


signals_with_noise = [get_signals_tests(db.ecg_noisy_signals[noisy_index-1], signal_dim, type="ecg noise",
                                        noisy_index=noisy_index) for noisy_index in range(1,5)]

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
