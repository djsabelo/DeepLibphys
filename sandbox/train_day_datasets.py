import time

import numpy as np

import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, segment_signal
from DeepLibphys.utils.functions.signal2model import Signal2Model

signal_dim = 64
hidden_dim = [16, 32, 64, 128, 256]
batch_size = 128
mini_batch_size = 16
window_size = 128
signal_directory = 'DAY_HRV_[128.256]'
n_for_each = 32

signals_tests = db.day_hf
# signals_models = db.signal_models

#   Load signals from rr database
all_signals = get_signals_tests(signals_tests, signal_dim)
# i = 1
hidden_dim = 128
signal_directory = 'DAY_HRV_HF_['+str(hidden_dim)+'.'+str(window_size)+']'
window_size
for group_signals in all_signals:
    model_name = 'day_hrv_hf_{0}'.format(i)
    signal2model = Signal2Model(model_name, signal_directory, signal_dim=signal_dim, window_size=window_size,
                          hidden_dim=hidden_dim, mini_batch_size=mini_batch_size, learning_rate_val=0.05, save_interval=500)
    model = GRU.LibphysMBGRU(signal2model)

    # model.load(dir_name=signal_directory, file_tag=model.get_file_tag(-5,-5))
    model.train_block(group_signals, signal2model, n_for_each=n_for_each, random_training=True)
#
# signal_dim = 64
# hidden_dim = 256
# batch_size = 128
# mini_batch_size = 16
# window_size = 64
# signal_directory = 'DAY_RR_[128.64]'
# n_for_each = 16
#
# signals_tests = db.day_vlf
# signals_models = db.signal_models
#
# all_signals = get_signals_tests(signals_tests, signal_dim)
# i = 0
# for group_signals in all_signals:
#     i += 1
#     model_name = 'day_hf_10Hz_{0}'.format(i)
#     signal2model = Signal2Model(model_name, signal_directory, signal_dim=signal_dim, window_size=window_size,
#                           hidden_dim=hidden_dim, mini_batch_size=mini_batch_size, learning_rate_val=0.01)
#     model = GRU.LibphysMBGRU(signal2model)
#     model.train_block(group_signals, signal2model, n_for_each=n_for_each, random_training=True)