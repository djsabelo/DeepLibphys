import time

import numpy as np

import DeepLibphys.models.libphys_MBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, segment_signal
from DeepLibphys.utils.functions.signal2model import Signal2Model


def train_block(signals, signal2model, signal_indexes, n_for_each):
    model = GRU.LibPhys_GRU(signal_dim=signal2model.signal_dim,
                            hidden_dim=signal2model.hidden_dim,
                            signal_name=signal2model.model_name,
                            n_windows=signal2model.mini_batch_size)
    model.save(signal2model.signal_directory, model.get_file_tag(-1, -1))

    x_train = []
    y_train = []
    for i in signal_indexes:
        X_windows, y_end_values, n_windows, last_index = segment_signal(signals[0][i], signal2model.window_size,
                                                                        overlap=0.33)
        Y_windows, y_end_values, n_windows, last_index = segment_signal(signals[1][i], signal2model.window_size,
                                                                        overlap=0.33)

        window_indexes = np.random.permutation(n_windows)  # randomly select windows
        if len(x_train) == 0:
            x_train = X_windows[window_indexes[0:n_for_each], :]
            y_train = Y_windows[window_indexes[0:n_for_each], :]
        else:
            x_train = np.append(x_train, X_windows[window_indexes[0:n_for_each], :], axis=0)
            y_train = np.append(y_train, Y_windows[window_indexes[0:n_for_each], :], axis=0)

        x_test = X_windows[window_indexes[n_windows:], :]
        y_test = Y_windows[window_indexes[n_windows:], :]

    model.save_test_data(model.get_file_tag(-5, -5), signal2model.signal_directory, [x_test, y_test])
    x_test = []
    y_test = []
    X_windows = []
    Y_windows = []
    t1 = time.time()
    model.train_with_msgd(x_train, y_train, signal2model.number_of_epochs, 0.9, track_loss=False,
                          save_directory=signal2model.signal_directory,
                          save_distance=signal2model.save_interval)
    print("Dataset trained in: ~%d seconds" % int(time.time() - t1))
    model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))


N = 5000
example_index = [7]
signal_dim = 64
hidden_dim = 256
# signal_name = "ecg_7"
# signal_model = "ecg_7"
signal_directory = 'BIOMETRIC_ECGs_[128.256]'
# learning_rate_val = 0.01
batch_size = 128
mini_batch_size = 16
window_size = 256
# number_of_epochs = 1000000

#Noisy signals
for noisy_index in [2]:#range(3,5):
    signals_tests = db.ecg_noisy_signals[noisy_index]
    signals_models = db.signal_models
    #
    # #   Load signals from database
    signals = get_signals_tests(signals_tests, signals_models[0].Sd, type="ecg noise", noisy_index=noisy_index)

    # train each signal from fantasia
    for i in range(9, 19):
        name = 'bio_noise_'+str(noisy_index)+'_ecg_' + str(i)
        signal = Signal2Model(name, signal_directory, batch_size=batch_size)
        model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=name, n_windows=mini_batch_size)
        model.save(signal_directory, model.get_file_tag(-1, -1))
        model.train_signals(signals[0][i], signals[1][i], signal, decay=0.95, track_loss=False)


# Normal + noisy ECGs
signal_dim = 64
hidden_dim = 256
signal_directory = 'BIOMETRIC_ECGs_[20.256]'
n_for_each = 16
mini_batch_size = n_for_each
signals_models = db.signal_models

signals_with_noise = [get_signals_tests(db.ecg_noisy_signals[noisy_index-1], signals_models[0].Sd, type="ecg noise",
                                        noisy_index=noisy_index) for noisy_index in range(1,5)]
signals_without_noise = get_signals_tests(db.signal_tests, signals_models[0].Sd, type="ecg")
signals = list(range(19))

for i in range(19):
    signals[i] = [[],[]]
    signals[i][0] = [signals_without_noise[0][i]] + [signals_with_noise[j][0][i] for j in range(4)]
    signals[i][1] = [signals_without_noise[0][i]] + [signals_with_noise[j][0][i] for j in range(4)]

# # train each signal from fantasia
# for i in range(0, 19):
#     name = 'biometry_with_noise_' + str(i)
#     signal2model = Signal2Model(name, signal_directory, mini_batch_size=mini_batch_size)
#     train_block(signals[i], signal2model, list(range(len(signals[0][0]))), n_for_each)
#
#             #
#     # signal = Signal2Model('ecg_7_history', signal_directory, batch_size=batch_size, number_of_epochs=2000)
#     # model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name='ecg_7_history', n_windows=mini_batch_size)
#     # model.save(signal_directory, model.get_file_tag(-1, -1))
#     #
#     # model.train_signal(signals[0][6], signals[1][6], signal, decay=0.95, track_loss=False, save_distance=10)



# n_for_each = 32
#
# # treinar todos os velhos mesmo tempo
# signal_indexes = list(range(0, 10))
# signal2model = Signal2Model('ecg_all_old', signal_directory,
#                             batch_size=len(signal_indexes) * n_for_each,
#                             mini_batch_size=mini_batch_size,
#                             learning_rate_val=0.1)
#
# train_block(signals, signal2model, signal_indexes, n_for_each)
#
# # treinar todos os novos
# signal_indexes = list(range(10, 19))
# signal2model = signal2model.signal_name = 'ecg_all_young'
#
# train_block(signals, signal2model, signal_indexes, n_for_each)

# train all fantasia dataset
# signal_indexes = list(range(1,19))
# n_for_each = 16
# signal2model = Signal2Model('ecg_all_fantasia', signal_directory, batch_size=len(signal_indexes) * n_for_each)
#
# train_block(signals, signal2model, signal_indexes, n_for_each)

########################################################################################################################
###########################################     EEG    #################################################################
# train each eeg from attention dataset

# signals = get_signals_tests(signals_tests, signals_models[0].Sd, type="eeg")
# signal_dim = 64
# hidden_dim = 256
# batch_size = 128
# mini_batch_size = 16
# window_size = 512
# signal_directory = 'EEGs_SEQ_ATTENTION_['+str(batch_size)+'.'+str(window_size)+']'
#
#
# for i in range(0,6):
#     print(signals[0][i])
#     name = 'eeg_attention' + str(i)
#     signal = Signal2Model(name, signal_directory, batch_size=batch_size, save_interval=1000)
#     model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=name, n_windows=mini_batch_size)
#     model.save(signal_directory, model.get_file_tag(-1, -1))
#     model.train_signals(signals[0][i], signals[1][i], signal, decay=0.95, track_loss=False, use_random=False)


########################################################################################################################
###########################################     GSR    #################################################################
# train each gsr from attention dataset

# signals = get_signals_tests(signals_tests, signals_models[0].Sd, type="gsr")
# signal_dim = 64
# hidden_dim = 256
# batch_size = 128
# mini_batch_size = 16
# window_size = 256
# signal_directory = 'DRIVER_GSR_['+str(batch_size)+'.'+str(window_size)+']'
# #
# # for i in range(len(signals[0])):
# #     print(signals[0][i])
# #     name = 'driver_gsr' + str(i)
# #     signal = Signal2Model(name, signal_directory, batch_size=batch_size)
# #     model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=name, n_windows=mini_batch_size)
# #     model.save(signal_directory, model.get_file_tag(-1, -1))
# #     model.train_signal(signals[0][i], signals[1][i], signal, decay=0.95, track_loss=False, save_distance=1000)
# #
# # # train driver gsr dataset
# signal_indexes = list(range(len(signals[0])))
# n_for_each = 32
# signal2model = Signal2Model('all_driver_gsr', signal_directory, batch_size=len(signal_indexes) * n_for_each)
#
# train_block(signals, signal2model, signal_indexes, n_for_each)


########################################################################################################################
###########################################     Biometric data    ######################################################
# train each eeg from attention dataset

# signals = get_signals_tests(signals_tests, signals_models[0].Sd[0].Sd, type="biometric")
# signal_dim = 64
# hidden_dim = 256
# batch_size = 128
# mini_batch_size = 16
# window_size = 256
# signal_directory = 'DRIVER_GSR_['+str(batch_size)+'.'+str(window_size)+']'
#
# for i in range(len(signals[0])):
#     print(signals[0][i])
#     name = 'driver_gsr' + str(i)
#     signal = Signal2Model(name, signal_directory, batch_size=batch_size)
#     model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=name, n_windows=mini_batch_size)
#     model.save(signal_directory, model.get_file_tag(-1, -1))
#     model.train_signal(signals[0][i], signals[1][i], signal, decay=0.95, track_loss=False, save_distance=1000)
#
# # train driver gsr dataset
# signal_indexes = list(len(signals[0]))
# n_for_each = 32
# signal2model = Signal2Model('all_driver_gsr', signal_directory, batch_size=len(signal_indexes) * n_for_each)
#
# train_block(signals, signal2model, signal_indexes, n_for_each)