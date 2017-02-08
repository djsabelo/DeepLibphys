import matplotlib.pyplot as plt

import BiosignalsDeepLibphys.utils.data.database as db
import BiosignalsDeepLibphys.utils.functions.libphys_RGRU as RGRU
from BiosignalsDeepLibphys.utils.functions.common import get_signals_tests
from BiosignalsDeepLibphys.utils.functions.signal2model import Signal2Model

# N = 5000
# example_index = [7]
hidden_dim = 1048
signal_name = "ecg_7"
signal_directory = 'ECGs_FANTASIA_[256.256]'
batch_size = 32
mini_batch_size = 8
window_size = 256
# number_of_epochs = 1000000

signals_tests = db.signal_tests
signals_models = db.signal_models

signal2model = Signal2Model(signal_name,
                            signal_directory,
                            hidden_dim=hidden_dim,
                            mini_batch_size=mini_batch_size,
                            window_size=window_size,
                            batch_size=batch_size,
                            save_interval=10000,
                            learning_rate_val=0.01)

model = RGRU.LibPhys_RGRU("RGRU_", hidden_dim=hidden_dim, mini_batch_dim=mini_batch_size)
signals = get_signals_tests(signals_tests, signals_models[0].Sd, index=7, regression=True)
plt.plot(signals[0][0])
plt.show()
model.train_signal(signals[0][0], signals[1][0], signal2model, save_distance=100, track_loss=True)
# model.train_signal(x, y, signal2model, save_distance=1000, track_loss=True)
