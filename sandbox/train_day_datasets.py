import time

import numpy as np

import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, segment_signal
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import DeepLibphys.models.LibphysSGDGRU as GRU

signal_dim = 64
hidden_dim = 64
mini_batch_size = 5
signal_directory = 'DAY_HRV_[ALL.256]'

signals_tests = db.day_rr
# signals_models = db.signal_models

#   Load signals from rr database
all_signals = get_signals_tests(signals_tests, signal_dim)

sizes = [[len(signal) for signal in signals] for signals in all_signals]
sizes = sizes[0] + sizes[1] + sizes[2] + sizes[3]
window_size = np.min(sizes) - 1

signal_directory = 'DAY_HRV_LOO_HF_['+str(hidden_dim)+'.'+str(window_size)+']'
z = 0
for i, group_signals in zip(range(z, len(all_signals)), all_signals[z:]):
    model_name = 'day_hrv_rr_{0}_{1}'.format(i, "-1")
    signal2model = Signal2Model(model_name, signal_directory,
                                signal_dim=signal_dim,
                                window_size=window_size,
                                hidden_dim=hidden_dim,
                                mini_batch_size=mini_batch_size,
                                learning_rate_val=0.1,
                                save_interval=100000)

    if i == 0:
        pass
    else:
        model = GRU.LibphysSGDGRU(signal2model)
        model.train_block(group_signals, signal2model, n_for_each=1)
    for j in range(len(group_signals)):
        if i == 0 and j < 29:
            pass
        else:
            train_group = group_signals
            train_group.pop(j)

            model_name = 'day_hrv_rr_{0}_{1}'.format(i, j)
            signal2model.model_name = model_name
            running_ok = False
            while not running_ok:
                model = GRU.LibphysSGDGRU(signal2model)
                running_ok = model.train_block(train_group, signal2model, n_for_each=1)




    # model.load(dir_name=signal_directory, file_tag=model.get_file_tag(-5,-5))

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