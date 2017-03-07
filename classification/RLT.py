# Relative Loss Threshold Classification

import numpy as np

import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, get_random_batch, randomize_batch, plot_confusion_matrix, segment_matrix
from DeepLibphys.utils.functions.database import ModelInfo
from DeepLibphys.utils.functions.signal2model import Signal2Model
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import time
import seaborn


def load_test_windows(signals_models, n_classification_windows=None):
    n_signals = len(signals_models)
    window_size = signals_models[0].W
    if n_classification_windows is None:
        x_test_len = [len(GRU.load_test_data(model_info.dataset_name, model_info.directory)[0])
                      for model_info in signals_models]
        n_classification_windows = min(x_test_len)

    X_matrix = np.zeros((n_signals, n_classification_windows, window_size))
    Y_matrix = np.zeros((n_signals, n_classification_windows, window_size))

    for i, model_info in zip(range(n_signals), signals_models):
        [x_test, y_test] = GRU.load_test_data(model_info.dataset_name, model_info.directory)
        X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(x_test, y_test, n_classification_windows)

    return X_matrix, Y_matrix


def get_segmented_test_signals(test_signals, window_size=256, n_classification_windows=None, overlap=0.33):
    n_signals = len(test_signals)

    test_signals_windows = [segment_matrix(test_signals, window_size, overlap) for test_signal in test_signals]
    if n_classification_windows is None:
        n_classification_windows = min([len(test_windows) for test_windows in test_signals_windows])

    X_matrix = np.zeros((n_signals, n_classification_windows, window_size))
    Y_matrix = np.zeros((n_signals, n_classification_windows, window_size))

    for i, test_signal in zip(range(n_signals), test_signals):
        X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(
            test_signals_windows[0], test_signals_windows[1], n_classification_windows)


def calculate_loss_tensor(signals_models_info, X_test_matrix, Y_test_matrix):
    print("Loading model...")
    model_info = signals_models_info[0]
    n_models = np.shape(signals_models_info)[0]
    n_signals = np.shape(X_test_matrix)[0]
    n_classification_windows = np.shape(X_test_matrix)[1]
    window_batch = n_classification_windows
    windows_indexes = [n_classification_windows]

    if int(n_classification_windows / 256) > 1:
        window_batch = int(n_classification_windows/4)
        X_test_matrix = X_test_matrix[:, 0:n_classification_windows, :]
        Y_test_matrix = Y_test_matrix[:, 0:n_classification_windows, :]
        windows_indexes = np.arange(0, n_classification_windows - window_batch + 1, window_batch)
        n_classification_windows = len(windows_indexes)*window_batch

    signal2model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                    hidden_dim=model_info.Hd, window_size=model_info.W, mini_batch_size=window_batch)
    model = GRU.LibphysMBGRU(signal2model)

    loss_tensor = np.zeros((n_models, n_signals, n_classification_windows))
    for m, model_info in zip(range(n_models), signals_models_info):
        model.signal_name = model_info.dataset_name
        model.load(dir_name=model_info.directory)
        print("Processing " + model_info.name)

        for s in range(n_signals):
            print("Calculating loss of Model {0} for Signal {1}".format(model_info.name, s), end=' - ')
            for w in windows_indexes:
                loss_tensor[m, s, w:w+window_batch] = np.asarray(
                    model.calculate_loss_vector(
                        X_test_matrix[s, w:w + window_batch, :],
                        Y_test_matrix[s, w:w + window_batch, :]))

def save_loss_tensor(filename, loss_tensor, signal_models):
    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signal_models)

    return loss_tensor