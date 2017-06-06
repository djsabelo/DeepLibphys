import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import DeepLibphys.models.LibphysCrossGRU as GRU
from DeepLibphys.utils.functions.signal2model import *


if __name__ == "__main__":
    signal = np.zeros((6, 2, 65))
    for i in range(np.shape(signal)[0]):
        signal[i, 0, :] = np.array(np.arange(65)*4/65, dtype=int)
        signal[i, 1, :] = np.array(np.arange(65)*2/65, dtype=int)

    sig2mod = Signal2Model("test_x", "[TEST_CROSS_GRU]", signal_dim=5, window_size=64, hidden_dim=3, mini_batch_size=1)
    model = GRU.LibphysCrossGRU(sig2mod)
    print("in")
    model.train_model(signal[:, :, :-1], signal[:, :, 1:], sig2mod)

