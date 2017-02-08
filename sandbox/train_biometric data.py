import time

import numpy as np

import DeepLibphys.models.libphys_MBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, segment_signal
from DeepLibphys.utils.functions.signal2model import Signal2Model


def train_block(signals, signal2model, signal_indexes, n_for_each):
    model = GRU.LibPhys_GRU(signal_dim=signal2model.signal_dim,
                            hidden_dim=signal2model.hidden_dim,
                            signal_name=signal2model.signal_name,
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

number_of_epochs = 10000
signal_dim = 64
hidden_dim = 256
batch_size = 128
mini_batch_size = 16
window_size = 256
signal_directory = 'BIO_ACC_[{0}.{1}]'.format(window_size, batch_size)


signals_tests = db.signal_tests
signals_models = db.signal_models

for i in range(178, 300):
    try:
        SIGNAL_BASE_NAME = "biometric_acc_x_"
        X_train, Y_train, X_test, Y_test = get_signals_tests(signals_tests, signals_models[0].Sd, type="biometric", index=i)
        signal_name = SIGNAL_BASE_NAME + str(i)
        signal_info = Signal2Model(signal_name, signal_directory,
                                   signal_dim=signal_dim,
                                   hidden_dim=hidden_dim,
                                   learning_rate_val=0.05,
                                   batch_size=batch_size,
                                   window_size=window_size,
                                   number_of_epochs=number_of_epochs,
                                   mini_batch_size=mini_batch_size)

        model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=signal_name,
                                n_windows=mini_batch_size)
        model.save(signal_directory, model.get_file_tag(-1, -1))

        model.train_signals(X_train[1], Y_train[1], signal_info, decay=0.95, track_loss=False, X_test=X_test[1], Y_test=Y_test[1])

        SIGNAL_BASE_NAME = "biometric_acc_y_"
        X_train, Y_train, X_test, Y_test = get_signals_tests(signals_tests, signals_models[0].Sd, type="biometric", index=i)
        signal_name = SIGNAL_BASE_NAME + str(i)
        signal_info = Signal2Model(signal_name, signal_directory,
                                   signal_dim=signal_dim,
                                   hidden_dim=hidden_dim,
                                   learning_rate_val=0.05,
                                   batch_size=batch_size,
                                   window_size=window_size,
                                   number_of_epochs=number_of_epochs,
                                   mini_batch_size=mini_batch_size)

        model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=signal_name,
                                n_windows=mini_batch_size)
        model.save(signal_directory, model.get_file_tag(-1, -1))

        model.train_signals(X_train[1], Y_train[1], signal_info, decay=0.95, track_loss=False, X_test=X_test[1], Y_test=Y_test[1])

        SIGNAL_BASE_NAME = "biometric_acc_z_"
        signal_name = SIGNAL_BASE_NAME + str(i)
        signal_info = Signal2Model(signal_name, signal_directory,
                                   signal_dim=signal_dim,
                                   hidden_dim=hidden_dim,
                                   learning_rate_val=0.05,
                                   batch_size=batch_size,
                                   window_size=window_size,
                                   number_of_epochs=number_of_epochs,
                                   mini_batch_size=mini_batch_size)

        model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=signal_name,
                                n_windows=mini_batch_size)
        model.save(signal_directory, model.get_file_tag(-1, -1))

        model.train_signals(X_train[2], Y_train[2], signal_info, decay=0.95, track_loss=False, X_test=X_test[2], Y_test=Y_test[2])
    except:
        pass

    #ONLY 30:
    # 0.320222222222 - 0.319578544061
    # 0.142168218914 - 0.142046211811