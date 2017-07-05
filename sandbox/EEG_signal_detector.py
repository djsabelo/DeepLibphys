import numpy as np

import DeepLibphys.models.libphys_MBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import get_signals_tests, get_random_batch, randomize_batch, plot_confusion_matrix
from DeepLibphys.utils.functions.database import ModelInfo

GRU_DATA_DIRECTORY = "../data/trained/"

def load_test_data(filetag=None, dir_name=None):
    print("Loading test data...")
    filename = GRU_DATA_DIRECTORY + dir_name + '/' + filetag + '_test_data.npz'
    npzfile = np.load(filename)
    return npzfile["test_data"]

def calculate_loss_tensor(filename, N_Windows, W, signals_models, signals_info, test_included=False):
    loss_tensor = np.zeros((len(signals_models), len(signals_info), N_Windows))
    N_Signals = len(signals_info)

    X_matrix = np.zeros((N_Signals, N_Windows, W))
    Y_matrix = np.zeros((N_Signals, N_Windows, W))

    if not test_included:
        signals = get_signals_tests(signals_info, signals_models[0].Sd)
        for i in range(N_Signals):
            [X_matrix[i, :, :], Y_matrix[i, :, :]] = get_random_batch(signals[0][i], signals[1][i], W, N_Windows)
    else:
        i = 0
        for model_info in signals_models:
            [x_test, y_test] = load_test_data("GRU_" + model_info.dataset_name + "[64.256.-1.-1.-1]"
                                              , model_info.directory)
            X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(x_test, y_test, N_Windows)
            i += 1



    history = []
    m = -1
    for m in range(len(signals_models)):
        model_info = signals_models[m]
        model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
                                n_windows=N_Windows)
        model.load(signal_name=model_info.name, filetag=model.get_file_tag(model_info.DS,
                                                                           model_info.t),
                   dir_name=model_info.directory)




        print("Processing " + model_info.name)

        for s in range(N_Signals):
            x_test = X_matrix[s, :, :]
            y_test = Y_matrix[s, :, :]
            signal_info = signals_info[s]
            print("Calculating loss for " + signal_info.name, end=';\n ')
            loss_tensor[m, s, :] = np.asarray(model.calculate_loss_vector(x_test, y_test))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models,
             signals_tests=signals_info)


def get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests, N_Windows, no_numbers=False):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i].name for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i].name for i in Sig]

    for i in Sig:
        # if signals_models[i].name is not "None":
        loss_tensor[:-1, i, :] = loss_tensor[:-1, i, :] / np.max(loss_tensor[:-1, i, :])

    predicted_matrix = np.argmin(loss_tensor[Mod][:, Sig, :], axis=0)

    sinal_predicted_matrix = np.zeros((len(Sig), len(Mod)))

    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j) / N_Windows

    return sinal_predicted_matrix, labels_model, labels_signals


def get_confusion_matrix(labels, signal_predicted_matrix):
    correct = 0
    wrong = 0
    N = 0
    models = list(range(np.shape(signal_predicted_matrix)[1]))
    confusion_tensor = np.zeros((len(models), 2, 2))
    rejection = np.zeros(len(models))
    N_Windows = np.sum(signal_predicted_matrix[:, 0])

    for i in range(np.shape(signal_predicted_matrix)[0]):
        values = signal_predicted_matrix[i, :]
        N += np.sum(values)
        correct += values[labels[i]]
        values = np.delete(values, labels[i])
        wrong += np.sum(values)
        # [TP,FN]
        # [FP,TN]
        for j in range(len(models)):
            values = signal_predicted_matrix[i, :]
            if labels[i] == j:
                confusion_tensor[j, 0, 0] += values[j]
                values = np.delete(values, j)
                confusion_tensor[j, 1, 0] += np.sum(values)
            else:
                confusion_tensor[j, 0, 1] += values[j]
                values = np.delete(values, j)
                confusion_tensor[j, 1, 1] += np.sum(values)

            rejection[j] += values[-1]

    return confusion_tensor, correct, wrong, rejection


def print_confusion(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=False):
    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)
    plot_confusion_matrix(sinal_predicted_matrix, labels_signals, labels_model, no_numbers, norm=True)  # , cmap=cmap)


def print_mean_loss(Mod, Sig, loss_tensor, signals_models, signals_tests):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i]["s_name"] for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i][-1] for i in Sig]

    mean_values_matrix = np.mean(loss_tensor, axis=2)

    sinal_predicted_matrix = np.zeros(len(Sig))

    # for i in range(np.shape(sinal_predicted_matrix)[0]):
    for j in range(np.shape(sinal_predicted_matrix)[0]):
        sinal_predicted_matrix[j] = mean_values_matrix[0, j]

    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)
    plot_confusion_matrix(sinal_predicted_matrix.T, labels_model, labels_signals)  # , cmap=cmap)


def classify_biosignals(filename, threshold=300, models_index=None, signals_index=None):
    npzfile = np.load(filename + ".npz")
    loss_tensor, signals_models, signals_tests = \
        npzfile["loss_tensor"], npzfile["signals_models"], npzfile["signals_tests"]

    if models_index is None:
        models_index = list(range(np.shape(loss_tensor)[0]))
    if signals_index is None:
        signals_index = list(range(np.shape(loss_tensor)[1]))

    signals_models = np.hstack((signals_models, ModelInfo(name="None", dataset_name="none")))

    loss_tensor = np.vstack(
        (loss_tensor, threshold * np.ones((1, np.size(loss_tensor, axis=1), np.size(loss_tensor, axis=2)))))

    sinal_predicted_matrix, signal_labels, model_labels = get_sinal_predicted_matrix(
        models_index, signals_index, loss_tensor, signals_models, signals_tests, N_Windows, no_numbers=True)

    print_confusion(sinal_predicted_matrix, model_labels, signal_labels, no_numbers=True)

# CONFUSION_TENSOR_[W,Z]
N_Windows = 200
W = 512
filename = "../data/validation/CONFUSION_EEG_ALL_[" + str(N_Windows) + "," + str(W) + "]"

signals_models = db.eeg_models
# signals_models = signals_models[-2:]
signals_info = db.signal_tests

if type is not None:
    signals_aux = []
    for signal_info in signals_info:
        if signal_info.type == "eeg":
            signals_aux.append(signal_info)
    signals_info = signals_aux

calculate_loss_tensor(filename, N_Windows, W, signals_models, signals_info, False)

models = list(range(len(signals_models)))
signals = list(range(len(signals_info)))
classify_biosignals(filename, models_index=models, signals_index=signals, threshold=0.5)

# signals_models = db.signal_generic_models
# signals_info = db.signal_tests
#
# # if type is not None:
# #     signals_aux = []
# #     for signal_info in signals_info:
# #         if signal_info.type == "ecg":
# #             signals_aux.append(signal_info)
# #     signals_info = signals_aux
#
# calculate_loss_tensor(filename, N_Windows, W, signals_models, signals_info, type="ecg")
# models = list(range(len(signals_models)))
# signals = list(range(len(signals_info)))
# classify_biosignals(filename, models_index=models, signals_index=signals, threshold=0.08)
