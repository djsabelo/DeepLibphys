import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn
from DeepLibphys.utils.functions.signal2model import *
from DeepLibphys.utils.functions.common import *
import DeepLibphys.models.LibphysCrossGRU as GRU
import theano

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='low'
# theano.config.compute_test_value = 'warn'


def load_data():
    ecg = get_fantasia_dataset(signal_dim, [1], dataset_dir=FANTASIA_ECG, peak_into_data=False)[0][0]
    resp = get_fantasia_dataset(signal_dim, [1], dataset_dir=FANTASIA_RESP, peak_into_data=False)[0][0]

    ecgs, _, _, _ = segment_signal(ecg, window_size)
    resps, _, _, _ = segment_signal(resp, window_size)

    signal = np.array([np.vstack((ecg, resp)) for ecg, resp in zip(ecgs, resps)])
    np.savez('../data/Cross_GRU/test_signal.npz', signal=signal)


if __name__ == "__main__":
    signal_dim = 16
    hidden_dim = 32
    window_size = 1024

    # load_data()
    signal = np.load('../data/Cross_GRU/test_signal.npz')['signal']
    sig2mod = Signal2Model("test_x", "[TEST_CROSS_GRU]", signal_dim=signal_dim, window_size=window_size,
                           hidden_dim=hidden_dim, mini_batch_size=8)
    model = GRU.LibphysCrossGRU(sig2mod)
    model.load(dir_name="[TEST_CROSS_GRU]", file_tag=model.get_file_tag(epoch=1000))
    print("in")
    model.train_model(np.array(signal[:32, :, :-1], dtype=np.int32), np.array(signal[:32, :, 1:], dtype=np.int32), sig2mod)
    model.save("[TEST_CROSS_GRU]", model.get_file_tag(-5, -5))

