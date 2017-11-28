import numpy as np

import DeepLibphys
import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.database import ModelInfo
from DeepLibphys.utils.functions.signal2model import Signal2Model, Signal
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import math
from itertools import repeat
import time
import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import DeepLibphys.classification.RLTC as RLTC

GRU_DATA_DIRECTORY = "../data/trained/"
VALIDATION_DIRECTORY = "../data/validation/CYBHI"


def load_test_data(filetag=None, dir_name=None):
    print("Loading test data...")
    filename = GRU_DATA_DIRECTORY + dir_name + '/' + filetag + '_test_data.npz'
    npzfile = np.load(filename)
    return npzfile["test_data"]


def make_noise_signals(full_paths, max_target_SNR=16):
    target_SNR_array = np.arange(max_target_SNR, max_target_SNR - 5, -1)
    N_SIGNALS, N_NOISE, N_SAMPLES = len(full_paths), len(target_SNR_array), 0
    processed_noise_array = np.zeros((N_NOISE, N_SIGNALS, len(sio.loadmat(full_paths[0])['val'][0])))
    SNR = np.zeros(N_NOISE)
    for i, file_path in zip(range(len(full_paths)), full_paths):
        print("Processing " + file_path)
        signal = sio.loadmat(file_path)['val'][0]
        signal = remove_moving_std(remove_moving_avg((signal - np.mean(signal)) / np.std(signal)))
        smoothed_signal = smooth(signal)

        SNR = int(calculate_signal_to_noise_ratio(signal, smoothed_signal))

        last_std = 0.0001
        j = 0
        for target_SNR in target_SNR_array:
            signal_with_noise, last_std = make_noise_vectors(signal, smoothed_signal, target_SNR, last_std=last_std)
            processed_noise_array[j, i, :] = process_dnn_signal(signal_with_noise, 64)
            j += 1

    return processed_noise_array, target_SNR_array


def calculate_signal_to_noise_ratio(signal, smoothed_signal):
    signal = signal[13000:19000]
    smoothed_signal = smoothed_signal[13000:19000]
    noise_signal = smoothed_signal - signal
    SMOOTH_AMPLITUDE = np.max(smoothed_signal) - np.min(smoothed_signal)
    NOISE_AMPLITUDE = np.mean(np.abs(noise_signal)) * 2
    SNR = (10 * np.log10(SMOOTH_AMPLITUDE / NOISE_AMPLITUDE))
    return SNR


def make_noise_vectors(signal, smoothed_signal, target_SNR, last_std=0.0001):
    SNR = int(calculate_signal_to_noise_ratio(signal, smoothed_signal))
    added_noise = np.random.normal(0, last_std, len(signal))
    signal_with_noise = np.array(np.array(signal) + added_noise)
    print("Processing SNR {0}".format(target_SNR))
    if SNR < target_SNR:
        signal = smoothed_signal

    while int(SNR * 100) != target_SNR * 100:

        added_noise = np.random.normal(0, last_std, len(signal))
        signal_with_noise = np.array(np.array(signal) + added_noise)
        SNR = calculate_signal_to_noise_ratio(signal_with_noise, smoothed_signal)

        if int(SNR * 100) > target_SNR * 100:
            last_std *= 2
        else:
            last_std *= 0.2

    return signal_with_noise, last_std


def calculate_loss_tensors(N_Windows, W, signals_models):
    N_Versions = len(signals_models)
    N_Signals = len(signals_models[0])
    loss_tensor = np.zeros((N_Versions, N_Signals, N_Signals, N_Windows))
    X_matrix = np.zeros((N_Versions, N_Signals, N_Windows, W))
    Y_matrix = np.zeros((N_Versions, N_Signals, N_Windows, W))

    i = 0
    for model_info in signals_models[0]:
        x_tests = []
        y_tests = []
        for version in range(N_Versions):
            [x_test, y_test] = load_test_data(
                "GRU_" + model_info.dataset_name + "[" + str(model_info.Sd) + "." + str(model_info.Hd) + ".-1.-1.-1]"
                , model_info.directory)
            x_tests.append(x_test)
            y_tests.append(y_test)
        X_matrix[:, i, :, :], Y_matrix[:, i, :, :] = randomize_batch(np.asarray(x_test), np.asarray(y_test), N_Windows)
    i += 1

    print("Loading base model...")
    model_info = signals_models[0][0]
    model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
                            n_windows=N_Windows)

    for m in range(N_Signals):
        for version in range(N_Versions):
            model_info = signals_models[version][m]
            model.signal_name = model_info.dataset_name
            model.load(signal_name=model_info.name, filetag=model.get_file_tag(model_info.DS,
                                                                               model_info.t),
                       dir_name=model_info.directory)
            print("Processing " + model_info.name)

            for s in range(N_Signals):
                x_test = X_matrix[version, s, :, :]
                y_test = Y_matrix[version, s, :, :]
                print("Calculating loss for " + signals_models[version][s].name, end=';\n ')
                loss_tensor[version, m, s, :] = np.asarray(model.calculate_loss_vector(x_test, y_test))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models,
             signals_tests=signals_info)

    return loss_tensor


def calculate_loss_tensor(filename, Total_Windows, W, signals_models, signals=None):
    n_windows = Total_Windows #if Total_Windows < 256 else 256

    windows = np.arange(0, Total_Windows - n_windows + 1, n_windows)
    N_Windows = len(windows)
    N_Models = len(signals_models)
    Total_Windows = int(N_Windows * n_windows)
    N_Signals = len(signals)

    loss_tensor = np.zeros((N_Models, N_Signals, Total_Windows))

    X_matrix = np.array([signal[:Total_Windows, :-1] for signal in signals])
    Y_matrix = np.array([signal[:Total_Windows, 1:] for signal in signals])

    print("Loading model...")
    model_info = signals_models[0]

    signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                hidden_dim=model_info.Hd, mini_batch_size=n_windows)
    model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2Model)

    for m, model_info in zip(range(len(signals_models)), signals_models):
        model.model_name = model_info.dataset_name
        model.load(dir_name=model_info.directory)
        print("Processing Model " + model_info.name)

        for s in range(N_Signals):
            print("Calculating loss for ECG " + str(s + 1), end=';\n ')
            for w in windows:
                # index = w * n_windows
                x_test = X_matrix[s, w:w + n_windows, :]
                y_test = Y_matrix[s, w:w + n_windows, :]
                loss_tensor[m, s, w:w + n_windows] = np.asarray(model.calculate_mse_vector(x_test, y_test))

        # if not os.path.isdir(os.path.dirname(filename + ".npz")):
        #     os.mkdir(os.path.dirname(filename + ".npz"))

        np.savez(filename + ".npz",
                 loss_tensor=loss_tensor,
                 signals_models=signals_models)

        # for m in range(len(signals_models)):
        #     model_info = signals_models[m]
        #     model.model_name = model_info.dataset_name
        #     model.load(dir_name=model_info.directory)
        #     print("Processing Model " + model_info.name)
        #
        #     for s in range(N_Signals):
        #         print("Calculating loss for ECG " + str(s + 1), end=';\n ')
        #         for w in windows:
        #             index = w * n_windows
        #             x_test = X_matrix[s, index:index + n_windows, :]
        #             y_test = Y_matrix[s, index:index + n_windows, :]
        #             loss_tensor[m, s, index:index + n_windows] = np.asarray(model.calculate_mse_vector(x_test, y_test))
        #
        # if not os.path.isdir(os.path.dirname(filename + ".npz")):
        #     os.mkdir(os.path.dirname(filename + ".npz"))
        #
        # np.savez(filename + ".npz",
        #          loss_tensor=loss_tensor,
        #          signals_models=signals_models)

    return loss_tensor


def interate_loss_calculus(model_info, m, model, X_matrix, Y_matrix, n_windows, windows, N_Signals, loss_tensor):
    model.model_name = model_info.dataset_name
    model.load(dir_name=model_info.directory)
    print("Processing Model " + model_info.name)

    for s in range(N_Signals):
        print("Calculating model's " + model_info.name + " loss for ECG " + str(s + 1), end=';\n ')
        for w in windows:
            index = w * n_windows
            x_test = X_matrix[s, index:index + n_windows, :]
            y_test = Y_matrix[s, index:index + n_windows, :]
            loss_tensor[m, s, index:index + n_windows] = np.asarray(model.calculate_mse_vector(x_test, y_test))


def try_calculate_loss_tensor(filename, Total_Windows, W, signals_models, signals=None, noisy_index=None):
    print("Loading model...")
    n_windows = 250
    modelx_info = signals_models[0]

    signal2Model = Signal2Model(modelx_info.dataset_name, modelx_info.directory, signal_dim=modelx_info.Sd,
                                hidden_dim=modelx_info.Hd, mini_batch_size=n_windows)
    model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2Model)

    windows = np.arange(int(Total_Windows / n_windows))
    N_Windows = len(windows)
    N_Models = len(signals_models)
    Total_Windows = int(N_Windows * n_windows)
    N_Signals = len(signals)

    loss_tensor = np.zeros((N_Models, N_Signals, Total_Windows))

    X_matrix = np.zeros((N_Signals, Total_Windows, W))
    Y_matrix = np.zeros((N_Signals, Total_Windows, W))

    i = 0
    first_test_index = int(len(signals[0]) * 0.33)
    for signal in signals:
        signal_test = segment_signal(signal[first_test_index:], 256, 0.33)
        X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(signal_test[0], signal_test[1], Total_Windows)

        i += 1

    pool = Pool(8)
    pool.starmap(interate_loss_calculus, zip(signals_models, range(len(signals_models)),
                                             repeat(model), repeat(X_matrix), repeat(Y_matrix), repeat(n_windows),
                                             repeat(windows), repeat(N_Signals),
                                             repeat(loss_tensor)))

    if not os.path.isdir(os.path.dirname(filename + ".npz")):
        os.mkdir(os.path.dirname(filename + ".npz"))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models)

    return loss_tensor


def calculate_fine_loss_tensor(filename, Total_Windows, W, signals_models, n_windows):
    windows = np.arange(int(Total_Windows / n_windows))
    N_Windows = len(windows)
    N_Signals = len(signals_models)
    Total_Windows = int(N_Windows * n_windows)

    loss_tensor = np.zeros((N_Signals, N_Signals, N_Windows))

    X_matrix = np.zeros((N_Signals, Total_Windows, W))
    Y_matrix = np.zeros((N_Signals, Total_Windows, W))

    i = 0
    for model_info in signals_models:
        [x_test, y_test] = load_test_data(
            "GRU_" + model_info.dataset_name + "[" + str(model_info.Sd) + "." + str(model_info.Hd) + ".-1.-1.-1]"
            , model_info.directory)
        X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(x_test, y_test, Total_Windows)
        i += 1

    print("Loading model...")

    model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
                            n_windows=n_windows)

    for m in range(len(signals_models)):
        model_info = signals_models[m]
        model.signal_name = model_info.dataset_name
        model.load(signal_name=model_info.name, filetag=model.get_file_tag(model_info.DS,
                                                                           model_info.t),
                   dir_name=model_info.directory)
        print("Processing " + model_info.name)

        for s in range(N_Signals):
            print("Calculating loss for " + signals_models[s].name, end=';\n ')

            for w in windows:
                index = w * n_windows
                x_test = X_matrix[s, index:index + n_windows, :]
                y_test = Y_matrix[s, index:index + n_windows, :]
                loss_tensor[m, s, w] = np.asarray(model.calculate_loss(x_test, y_test))

    np.savez(filename + "_fine.npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models)

    return loss_tensor


def get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests=None, no_numbers=False):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i].name for i in Mod]
    # labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i].name for i in Sig]

    return calculate_classification_matrix(loss_tensor[Mod][:, Sig, :]), labels_model, labels_model


def calculate_classification_matrix(loss_tensor):
    normalized_loss_tensor = np.zeros_like(loss_tensor)
    for i in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:, i, :] = loss_tensor[:, i, :] - np.min(loss_tensor[:, i, :], axis=0)
        normalized_loss_tensor[:, i, :] = normalized_loss_tensor[:, i, :] / np.max(normalized_loss_tensor[:, i, :],
                                                                                   axis=0)

    predicted_matrix = np.argmin(normalized_loss_tensor, axis=0)

    sinal_predicted_matrix = np.zeros((np.shape(loss_tensor)[0], np.shape(normalized_loss_tensor)[1]))

    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j)  # / N_Windows

    return sinal_predicted_matrix


def normalize_tensor(loss_tensor):
    normalized_loss_tensor = np.zeros_like(loss_tensor)
    for j in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:, j, :] = loss_tensor[:, j, :] - np.min(loss_tensor[:, j, :], axis=0)
        normalized_loss_tensor[:, j, :] = normalized_loss_tensor[:, j, :] / np.max(loss_tensor[:, j, :], axis=0)

    return normalized_loss_tensor


def calculate_prediction_matrix(loss_tensor, threshold=1):
    threshold_layer = np.ones((np.shape(loss_tensor)[1], np.shape(loss_tensor)[2])) * threshold
    predicted_tensor = np.zeros_like(loss_tensor)
    normalized_loss_tensor = np.zeros(
        (np.shape(loss_tensor)[0] + 1, np.shape(loss_tensor)[1], np.shape(loss_tensor)[2]))
    normalized_loss_tensor[:-1, :, :] = normalize_tensor(loss_tensor)
    # normalized_loss_tensor[:-1,:,:] = loss_tensor
    normalized_loss_tensor[-1, :, :] = threshold_layer
    # print("\nThreshold "+str(threshold), end=":")

    # plt.figure("LOSS NORMALIZED")
    # plt.plot(normalized_loss_tensor[:, 1, 0])
    # plt.plot(normalized_loss_tensor[:, 1, 0])
    # plt.show()

    for i in range(np.shape(loss_tensor)[1]):
        predicted_tensor[i, :, :] = np.argmin(normalized_loss_tensor[[-1, i], :, :], axis=0)

    predicted_matrix = np.argmin(normalized_loss_tensor, axis=0)

    return predicted_matrix, predicted_tensor


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


def get_signal_confusion_matrix(signal_predicted_matrix):
    signal_list = list(range(np.shape(signal_predicted_matrix)[0]))
    confusion_tensor = np.zeros((len(signal_list), 2, 2))
    N_Windows = np.sum(signal_predicted_matrix[:, 0])
    rejection_rate = np.zeros_like(signal_list)
    rejection_signal = np.shape(signal_predicted_matrix)[0]

    for signal in signal_list:
        # [TP,FN]
        # [FP,TN]
        confusion_tensor[signal, 0, 0] = len(np.where(signal_predicted_matrix[signal, :] == signal)[0])  # TP
        confusion_tensor[signal, 0, 1] = len(np.where(signal_predicted_matrix[signal, :] != signal)[0])  # FN

        confusion_tensor[signal, 1, 0] = len(
            np.where(signal_predicted_matrix[np.arange(np.shape(signal_predicted_matrix)[0]) != signal, :] == signal)[
                0])  # FP
        confusion_tensor[signal, 1, 1] = len(
            np.where(signal_predicted_matrix[np.arange(np.shape(signal_predicted_matrix)[0]) != signal, :] != signal)[
                0])  # TN

        rejection_rate[signal] = len(np.where(signal_predicted_matrix[signal, :] == rejection_signal)[0]) / N_Windows

    return confusion_tensor, rejection_rate


def get_classification_confusion(signal_predicted_tensor):
    signal_list = np.arange(np.shape(signal_predicted_tensor)[0])
    confusion_tensor = np.zeros((len(signal_list), 2, 2))
    N_Windows = np.sum(signal_predicted_tensor[:, 0])
    rejection_rate = np.zeros_like(signal_list)
    rejection_signal = np.shape(signal_predicted_tensor)[0]

    for signal in signal_list:
        # [TP,FN]
        # [FP,TN]
        classified_matrix = signal_predicted_tensor[signal, :, :]

        confusion_tensor[signal, 0, 0] = len(np.where(classified_matrix[signal, :] == 1)[0])  # TP
        confusion_tensor[signal, 0, 1] = len(np.where(classified_matrix[signal, :] == 0)[0])  # FN

        confusion_tensor[signal, 1, 0] = len(
            np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 1)[0])  # FP
        confusion_tensor[signal, 1, 1] = len(
            np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 0)[0])  # TN

    # print(confusion_tensor[0,:,:])
    return confusion_tensor


def print_confusion(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=False, norm=True, title=""):
    print(sinal_predicted_matrix)

    plot_confusion_matrix(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=no_numbers,
                          norm=norm, title=title)
    # cmap = make_cmap(get_color(), max_colors=1000)


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


def classify_biosignals(filename, N_Windows=None, models_index=None, signals_index=None, w_for_classification=1,
                        title=""):
    npzfile = np.load(filename + ".npz")
    loss_tensor, signals_models = \
        npzfile["loss_tensor"], npzfile["signals_models"]

    if models_index is None:
        models_index = list(range(np.shape(loss_tensor)[0]))
    if signals_index is None:
        signals_index = list(range(np.shape(loss_tensor)[1]))

    signals_models = np.hstack((signals_models, ModelInfo(name="None", dataset_name="none")))

    # loss_tensor = np.vstack(
    #     (loss_tensor, threshold * np.ones((1, np.size(loss_tensor, axis=1), np.size(loss_tensor, axis=2)))))

    if w_for_classification > 1:
        temp_loss_tensor = calculate_min_windows_loss(loss_tensor, w_for_classification)
    else:
        temp_loss_tensor = loss_tensor

    N_Windows = np.shape(temp_loss_tensor)[2]

    sinal_predicted_matrix, signal_labels, model_labels = get_sinal_predicted_matrix(
        models_index, signals_index, temp_loss_tensor, signals_models, N_Windows, no_numbers=True)

    print_confusion(sinal_predicted_matrix, model_labels, signal_labels, no_numbers=False, norm=True, title=title)


def calculate_variables(loss_tensor, threshold=0.1, index=None):
    prediction_matrix, prediction_tensor = calculate_prediction_matrix(loss_tensor, threshold)
    N_Signals = np.shape(prediction_matrix)[0]
    N_Windows = np.shape(prediction_matrix)[1]
    probabilities = np.zeros_like(prediction_matrix)

    # Probability distribution
    # for i in range(N_Signals):
    #     for j in range(N_Signals):
    #         # probability of i being j
    #         probabilities[i,j] = (len(np.where(prediction_matrix[i,:] == j)[0]) / N_Windows)
    #         print(prediction_matrix[i,:])
    #
    #     plt.figure("Signal #"+str(i))
    #     plt.hist(prediction_matrix[i,:])
    #
    # plt.show()

    #

    confusion_tensor = get_classification_confusion(prediction_tensor)
    N_signals = np.shape(prediction_matrix)[0]
    labels = list(range(N_signals))

    # i = 0
    # label = labels[i]
    # prediction = prediction_matrix[:,i]
    # print(labels)
    # print(confusion_tensor[0,:,:])
    scores = list(range(N_Signals))

    # [TP,FN]
    # [FP,TN]
    for i in list(range(N_Signals)):
        # TRUE POSITIVE
        TP = confusion_tensor[i, 0, 0]
        # TRUE NEGATIVE
        TN = confusion_tensor[i, 1, 1]
        # FALSE POSITIVE
        FP = confusion_tensor[i, 1, 0]
        # FALSE NEGATIVE
        FN = confusion_tensor[i, 0, 1]

        # SENSITIVITY - TRUE POSITIVE RATE
        TPR = TP / (TP + FN)

        # SPECIFICITY - TRUE NEGATIVE RATE
        TNR = TN / (TN + FP)

        # PRECISION - POSITIVE PREDICTIVE VALUE
        PPV = TP / (TP + FP)

        # NEGATIVE PREDICTIVE VALUE
        NPV = TN / (TN + FN)

        # FALL-OUT - FALSE POSITIVE RATE
        FPR = FP / (FP + TN)

        # FALSE DISCOVERY RATE
        FDR = FP / (FP + TP)

        # MISS RATE - FALSE NEGATIVE RATE
        FNR = FN / (FN + TP)

        # ACCURACY
        ACC = (TP + TN) / (TP + TN + FP + FN)

        # F1 SCORE
        F1 = 2 * TP / (2 * TP + FP + FN)

        scores[i] = [FNR, FPR, TPR, ACC]

        # print(scores[i]["TPR"])
        # if i ==0:
        #     print(confusion_tensor[i])
        #     print("FPR - "+str(scores[i]["TPR"]))
        #     print("FNR - "+str(scores[i]["FNR"]))

    return scores


def calculate_roc(loss, step=0.001, last_index=1, first_index=0):
    last_index += step
    first_index -= step
    N_Signals = np.shape(loss)[0]
    eer = np.zeros((2, N_Signals)) - 1

    thresholds = np.arange(first_index, last_index + step, step)
    n_thresholds = len(thresholds)

    end_roc1 = np.vstack((np.zeros((1, np.shape(loss)[0])),
                          np.ones((1, np.shape(loss)[0]))))
    end_roc2 = np.ones((2, np.shape(loss)[0]))
    roc1 = np.vstack((np.zeros((1, n_thresholds, np.shape(loss)[0])),
                      np.ones((1, n_thresholds, np.shape(loss)[0]))))
    roc2 = np.ones((2, n_thresholds, np.shape(loss)[0]))
    scores = []
    acc = np.zeros((n_thresholds, np.shape(loss)[0] + 1))
    acc[:, -1] = thresholds
    for i in range(n_thresholds):
        if i % 100 == 0:
            print(".", end="")
        score = calculate_variables(loss, thresholds[i])
        for j in range(np.shape(loss)[0]):
            roc1[0, i, j] = score[j]["FNR"]
            roc1[1, i, j] = score[j]["FPR"]

            roc2[0, i, j] = score[j]["TPR"]
            roc2[1, i, j] = score[j]["FPR"]
            acc[i, j] = score[j]["ACC"]

        if np.logical_and(np.alltrue(roc1[:, i, :] == end_roc1), np.alltrue(roc2[:, i, :] == end_roc2)):
            break
        elif np.logical_and(len(np.where(np.not_equal(roc1[:, i, :], end_roc1))[0]) == 1,
                            len(np.where(np.not_equal(roc2[:, i, :], end_roc2))[0] == 1)):
            roc1[:, i:-1, :] = roc1[:, i, :] + np.zeros_like(roc1[:, i:-1, :])
            roc2[:, i:-1, :] = roc2[:, i, :] + np.zeros_like(roc2[:, i:-1, :])
            acc[i:-1, :] = acc[i, :] + np.zeros_like(acc[i:-1, :])
            break


            # print("ACC {0} - "+str(scores[0]["ACC"]), end =";")
            # print("TP {0} - " + str(scores[0]["TP"]), end=";")
            # print("FP {0} - " + str(scores[0]["FP"]), end=";")
            # print("FN {0} - " + str(scores[0]["FN"]), end=";")
            # print("TN {0} - " + str(scores[0]["TN"]))
        scores.append(score)
    remake = False
    candidate_index = 1
    print("end")
    candidate_index = 0
    for j in range(np.shape(loss)[0]):
        # non_zeros = np.where(roc1[0, :, j] > 0)
        # non_zeros1 = np.where(roc2[0, :, j] > 0)
        # non_zeros = np.unique(np.append(np.squeeze(non_zeros), np.squeeze(non_zeros1)))
        candidate = roc1[0, :, j] - roc1[1, :, j]
        candidate_index = np.argmin(candidate[candidate > 0])
        eer[0, j] = roc1[0, candidate_index, j]
        eer[1, j] = roc1[1, candidate_index, j]

        if candidate[candidate_index] != 0:
            eer_j, new_fpr_x, new_fpr_y = find_eeq(roc1, j)
            eer[0, j] = eer_j
            eer[1, j] = eer_j
            # print(eer_j)



            # if remake and step > 0.0000001:
            # plot_roc(roc1, roc2, eer)
            #     print("I have to remake.... taking more time....{0}".format(x[candidate_index]))
            #     maxi = candidate_index+100
            #     mini = candidate_index-100
            #     if maxi > len(x):
            #         maxi = -1
            #     if mini <= 0:
            #         mini = 0

            # roc1, roc2, scores, eer = calculate_roc(loss_tensor, step = step*0.01, last_index=x[maxi], first_index=x[mini])

    return roc1, roc2, scores, eer, acc, acc[np.argmin(candidate[candidate > 0])]


def calculate_smart_roc(loss, last_index=1, first_index=0):
    last_index += 1
    first_index -= 1
    thresholds = np.unique(np.round(normalize_tensor(loss), 4))
    thresholds = np.insert(thresholds, [0, len(thresholds)], [first_index, last_index])
    n_thresholds = len(thresholds)

    N_Signals = np.shape(loss)[0]
    eer = np.zeros((2, N_Signals)) - 1
    end_roc1 = np.vstack((np.zeros((1, np.shape(loss)[0])),
                          np.ones((1, np.shape(loss)[0]))))
    end_roc2 = np.ones((2, np.shape(loss)[0]))
    roc1 = np.vstack((np.zeros((1, n_thresholds, np.shape(loss)[0])),
                      np.ones((1, n_thresholds, np.shape(loss)[0]))))
    roc2 = np.ones((2, n_thresholds, np.shape(loss)[0]))
    scores = []

    for i in range(n_thresholds):
        if i % 100 == 0:
            print(".", end="")
        score = calculate_variables(loss, thresholds[i])
        for j in range(np.shape(loss)[0]):
            roc1[0, i, j] = score[j][0]
            roc1[1, i, j] = score[j][1]

            roc2[0, i, j] = score[j][2]
            roc2[1, i, j] = score[j][1]

            # print("ACC {0} - "+str(scores[0]["ACC"]), end =";")
            # print("TP {0} - " + str(scores[0]["TP"]), end=";")
            # print("FP {0} - " + str(scores[0]["FP"]), end=";")
            # print("FN {0} - " + str(scores[0]["FN"]), end=";")
            # print("TN {0} - " + str(scores[0]["TN"]))
        scores.append(score)
    remake = False
    candidate_index = 1
    print("end")
    candidate_index = 0
    for j in range(np.shape(loss)[0]):
        # non_zeros = np.where(roc1[0, :, j] > 0)
        # non_zeros1 = np.where(roc2[0, :, j] > 0)
        # non_zeros = np.unique(np.append(np.squeeze(non_zeros), np.squeeze(non_zeros1)))
        candidate = roc1[0, :, j] - roc1[1, :, j]
        candidate_index = np.argmin(candidate[candidate > 0])
        eer[0, j] = roc1[0, candidate_index, j]
        eer[1, j] = roc1[1, candidate_index, j]

        if candidate[candidate_index] != 0:
            eer_j, new_fpr_x, new_fpr_y = find_eeq(roc1, j)
            eer[0, j] = eer_j
            eer[1, j] = eer_j
            # print(eer_j)

    return roc1, roc2, scores, eer, thresholds, candidate_index


def plot_roc(roc1, roc2, eer, N_Signals=20):
    N_Signals = np.shape(loss_tensor)[1]
    name = 'gnuplot'
    cmap = plt.get_cmap(name)
    cmap_list = [cmap(i) for i in np.linspace(0, 1, N_Signals)]
    fig_1 = "ROC False Negative Rate/False Positive Rate"
    fig_2 = "ROC True Positive Rate/False Positive Rate"
    for j in range(np.shape(loss_tensor)[0]):
        plt.figure(fig_1)
        plt.scatter(roc1[1, :, j], roc1[0, :, j], marker='.', color=cmap_list[j])
        plt.plot(roc1[1, :, j], roc1[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal {0}'.format(j))

        plt.figure(fig_2)
        plt.scatter(roc2[1, :, j], roc2[0, :, j], marker='.', color=cmap_list[j])
        plt.plot(roc2[1, :, j], roc2[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal {0}'.format(j))

    N_Signals = np.shape(eer)[1]
    for signal in range(N_Signals):
        plt.figure(fig_1)
        plt.plot(eer[1, signal], eer[0, signal], color="#990000", marker='o', alpha=0.2)

    plt.figure(fig_1)
    plt.plot([0, 1], [0, 1], color="#990000", alpha=0.2)

    plt.ylabel("False Negative Rate")
    plt.xlabel("False Positive Rate")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()

    plt.figure(fig_2)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend()
    plt.show()


def get_min_max(j, interval, roc):
    """

    :param j:
    :param interval:
    :param roc:
    :return:
    """
    diff_roc = roc[0, :, j] - roc[1, :, j]
    if (len(np.where(diff_roc < 0)[0]) == 0):
        index = 0
    else:
        index = np.where(diff_roc < 0)[0][0]

    # if index == np.shape(roc)[1]-1:
    #     return -1

    min_max = [index - interval, index + interval]

    if min_max[0] < 0:
        min_max[1] = interval
        min_max[0] = 0
    elif min_max[1] >= len(roc[0, :, j]):
        min_max[0] -= (len(roc[0, :, j]) + 1 + interval)
        min_max[1] = len(roc[0, :, j]) - 1
        if min_max[0] < 0:
            min_max[1] = interval
            min_max[0] = 0

    return min_max


def find_eeq(roc, j):
    interval = 10
    min_max = get_min_max(j, interval, roc)
    roc_min_max = roc[:, min_max, j]

    while (roc_min_max[0, 0] - roc_min_max[0, 1] == 0) or (roc_min_max[1, 0] - roc_min_max[1, 1] == 0):
        interval += 2
        if interval >= np.shape(roc)[1]:
            return 0, roc[0, -1, j], roc[1, -1, j]
        min_max = get_min_max(j, interval, roc)
        roc_min_max = roc[:, min_max, j]

    array_indexes = list(range(min_max[0], min_max[1]))
    # fpr_x, new_array_indexes = np.unique(roc[1, array_indexes, j])
    fnr_y = roc[0, array_indexes, j]
    fpr_x = roc[1, array_indexes, j]

    x_interpolation = interpolate.interp1d(array_indexes, fpr_x)
    y_interpolation = interpolate.interp1d(array_indexes, fnr_y)

    candidate = np.array([1])
    candidate_index = np.argmin(candidate)
    array_size = 50
    new_fnr_y = []
    patience = 0
    PATIENCE_MAX = 20
    while (candidate[candidate_index] > 0.00001 and patience < PATIENCE_MAX):
        patience += 1
        new_array_indexes = np.linspace(array_indexes[0], array_indexes[-1], num=array_size)
        try:
            new_fpr_x = x_interpolation(new_array_indexes)
            new_fnr_y = y_interpolation(new_array_indexes)

            # print(new_fnr_y)
            candidate = abs(new_fpr_x - new_fnr_y)
            candidate_index = np.argmin(candidate)
            # print(new_fnr_y[candidate_index], end="..")
            # if len(new_array_indexes) != len(array_indexes):

            array_size *= 2
        except:
            plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
            plt.scatter(fpr_x, fnr_y, marker='.')
            plt.show()
            break

    # plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
    # plt.scatter(fpr_x, fnr_y, marker='.')
    # plt.show()
    eer = new_fnr_y[candidate_index]
    if patience == PATIENCE_MAX:
        print("Patience is exausted when searching for eer, minimum found: " + str(candidate[candidate_index]))
    if math.isnan(eer):
        print(new_fpr_x)
        print(new_fnr_y)
    # print(eer)
    return eer, new_fpr_x, new_fnr_y


def plot_EERs(EERs, time, labels, title="", file="_iterations", savePdf=False):
    cmap = matplotlib.cm.get_cmap('rainbow')
    fig = plt.figure("fig", figsize=(900 / 96, 600 / 96), dpi=96)
    for i, EER in zip(range(len(EERs)), EERs):
        plt.plot(time, EER, '.', color=cmap(i / len(EERs)), alpha=0.2)
        plt.plot(time, EER, color=cmap(i / len(EERs)), alpha=0.2, label=labels[i])

    mean_EERs = np.mean(EERs, axis=0).squeeze()
    index_min = np.argmin(mean_EERs)
    plt.plot(time, mean_EERs, 'b.', alpha=0.5)
    plt.plot(time, mean_EERs, 'b-', alpha=0.5, label="Mean")
    plt.plot(time[index_min], np.min(mean_EERs), 'ro', alpha=0.6)
    indexi = 0.6 * np.max(time)
    indexj = 0.5 * (np.max(EER) - np.min(EER))

    step = (np.max(EER) - np.min(EER))
    plt.annotate("EER MIN MEAN = {0:.4f}".format(mean_EERs[index_min]),
                 xy=(indexi, indexj))

    plt.annotate("EER MIN STD = {0:.4f}".format(np.std(EER)),
                 xy=(indexi, indexj - step))

    plt.annotate("TIME FOR MIN MEAN = {0:.4f}".format(time[index_min]),
                 xy=(indexi, indexj - step * 2))

    plt.annotate("TIME FOR MIN STD = {0:.4f}".format(mean_EERs[index_min]),
                 xy=(indexi, indexj - step * 3))

    plt.legend()
    plt.title(title)

    if savePdf:
        print("Saving img_2/EER{0}.pdf".format(file))
        pdf = PdfPages("img_2/EER{0}.pdf".format(file))
        pdf.savefig(fig)
        plt.clf()
        pdf.close()
    else:
        plt.show()
    return plt


def process_EER(loss_tensor, iterations=10, savePdf=True, SNR=1, name=""):
    all_data = []
    batch_size_array = np.arange(1, 60)
    N_Signals = np.shape(loss_tensor)[0]
    thresh = []
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = batch_size_array * 0.33
    for iteration in range(iterations):
        for batch_size in batch_size_array:
            temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
            if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
                break

                roc1, roc2, scores, eer, thresholds, candidate_index = calculate_smart_roc(temp_loss_tensor)
                thresh.append
            EERs[:, batch_size - 1, iteration] = eer[0, :]

            # plot_roc(roc1, roc2, eer_min)
            print("EER MIN: ,{0}".format(eer[0]))

            all_data.append(roc1, roc2, scores, eer, thresholds, candidate_index)

        labels = ["ECG {0}".format(i) for i in range(1, N_Signals + 1)]

        plot_EERs(EERs[:, :, iteration], seconds, labels, "Mean EER for SNR of {0}".format(SNR),
                  "_SNR_{0}{1}_{2}".format(SNR, name, iteration), savePdf=savePdf)

    np.savez(VALIDATION_DIRECTORY + "/ALL_DATA_SNR_{0}{1}.npz".format(SNR, name), all_data=all_data)
    labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    plot_EERs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations",
              "_SNR_ITER_{0}{1}".format(SNR, name), savePdf=savePdf)

    return np.mean(EERs, axis=0).T, thresholds


def calculate_min_windows_loss(loss_tensor, batch_size):
    N_Models = np.shape(loss_tensor)[0]
    N_Signals = np.shape(loss_tensor)[1]
    N_Total_Windows = np.shape(loss_tensor)[2]
    batch_indexes = np.arange(0, N_Total_Windows - batch_size, batch_size)
    N_Batches = len(batch_indexes)
    if N_Batches == 0:
        return -1

    temp_loss_tensor = np.zeros((N_Models, N_Signals, N_Batches))
    print(batch_size, end=" - ")
    randomized_loss_tensor = loss_tensor
    x = 0
    for i in batch_indexes:
        loss_batch = randomized_loss_tensor[:, :, i:i + batch_size]
        temp_loss_tensor[:, :, x] = np.min(loss_batch, axis=2)
        x += 1

    return temp_loss_tensor


def calculate_max_windows_loss(loss_tensor, batch_size):
    N_Signals = np.shape(loss_tensor)[1]
    N_Models = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    batch_indexes = np.arange(0, N_Total_Windows - batch_size, batch_size)
    N_Batches = len(batch_indexes)
    if N_Batches == 0:
        return -1

    temp_loss_tensor = np.zeros((N_Models, N_Signals, N_Batches))
    print(batch_size, end=" - ")

    loss_tensor = loss_tensor[:, :, np.random.permutation(N_Total_Windows)]
    x = 0
    for i in batch_indexes:
        loss_batch = loss_tensor[:, :, i:i + batch_size]
        temp_loss_tensor[:, :, x] = np.max(loss_batch, axis=2)
        x += 1

    return temp_loss_tensor


def calculate_mean_windows_loss(loss_tensor, batch_size):
    N_Signals = np.shape(loss_tensor)[1]
    N_Models = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    batch_indexes = np.arange(0, N_Total_Windows - batch_size, batch_size)
    N_Batches = len(batch_indexes)
    if N_Batches == 0:
        return -1

    temp_loss_tensor = np.zeros((N_Models, N_Signals, N_Batches))
    print(batch_size, end=" - ")

    loss_tensor = loss_tensor[:, :, np.random.permutation(N_Total_Windows)]
    x = 0
    for i in batch_indexes:
        loss_batch = loss_tensor[:, :, i:i + batch_size]
        temp_loss_tensor[:, :, x] = np.mean(loss_batch, axis=2)
        x += 1

    return temp_loss_tensor


def calculate_mean_error(loss_tensor):
    loss_tensor = loss_tensor / np.sum(loss_tensor, axis=1)
    return np.mean(loss_tensor, axis=2)


# CONFUSION_TENSOR_[W,Z]
# N_Windows = 6000

def calculate_all_loss_tensors(all_models_info, all_signals, filenames, N_Windows=6000, W=256):
    loss_quartenion = []

    if N_Windows is None:
        first_index = int(0.33 * len(all_signals[0, 0]))
        _, _, N_Windows, _ = segment_signal(all_signals[0, 0], W, 0.33, None, first_index)

    for models_info, filename, signals in zip(all_models_info, filenames, all_signals):
        loss_quartenion.append(calculate_loss_tensor(filename, N_Windows, W, models_info, signals))

    return loss_quartenion


def calculate_eers(batch_size, loss_tensor, iteration):
    print("batch_size = {0}, iteration = {1}".format(batch_size, iteration))
    temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
    if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
        return

    roc1, roc2, scores, eer, thresholds, candidate_index = calculate_smart_roc(temp_loss_tensor)
    # plot_roc(roc1, roc2, eer_min)
    print("EER MIN of batch_size = {0}, iteration = {1}: {2}, n windows = {3}"
          .format(batch_size, iteration, np.min(eer), np.shape(temp_loss_tensor)[2]))

    return [roc1, roc2, scores, eer, thresholds, candidate_index]


def descompress_data(data):
    roc1, roc2, scores, eer, thresholds, candidate_index = [], [], [], [], [], []
    for line in data:
        roc1.append(line[0])
        roc2 = line[1]
        scores.append(line[2])
        eer.append(line[3])
        thresholds.append(line[4])
        candidate_index.append(line[5])

    EERs = np.array(eer).T
    new_data = {
        "EERs": EERs,
        "scores": scores,
        "roc1": roc1,
        "roc2": roc2,
        "best": np.argmin(np.mean(EERs, axis=0)),
        "thresholds": thresholds,
        "candidate_index": candidate_index
    }
    return roc1, roc2, scores, eer, thresholds, candidate_index, new_data


def process_alternate_eer(loss_tensor, iterations=10, savePdf=True, SNR=None, name="", batch_size=120, fs=250,
                          labels=None):
    all_data = []
    batch_size_array = np.arange(1, batch_size)
    N_Signals = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = batch_size_array * 0.11 * W / fs
    all_data = []
    thresh = []
    for iteration in range(iterations):
        pool = Pool(10)
        x = np.round(time.time() * 10 ** 7) * iteration / np.random.randint(10 ** 7, (10 ** 8 - 1), 1)
        if x < 2 ** 32 - 1:
            np.random.seed(int(x))
        data = pool.starmap(calculate_eers, zip(batch_size_array, repeat(loss_tensor), repeat(iteration)))
        pool.close()
        # data = [calculate_eers(batch, loss_tensor, iteration) for batch in batch_size_array]
        roc1, roc2, scores, eer, thresholds, candidate_index, new_data = descompress_data(data)
        EERs[:, :, iteration] = np.array(eer)[:, 1, :].T
        all_data.append(data)
        if labels is None:
            labels = ["ECG {0}".format(i) for i in range(1, N_Signals + 1)]
        if SNR is None:
            plot_errs(EERs[:, :, iteration], seconds, labels, VALIDATION_DIRECTORY, "EER_{0}".format(name), "Mean EER",
                      savePdf=savePdf)
        else:
            plot_errs(EERs[:, :, iteration], seconds, labels, VALIDATION_DIRECTORY, "EER_{0}".format(name),
                      "Mean EER for SNR of {0}".format(SNR), savePdf=savePdf)
            # plot_errs(accs[:-1, :, iteration], seconds, labels, "Mean ACC for SNR of {0}".format(SNR),
            #           "_ACC_SNR_{0}_{1} and Threshold of {2}".format(SNR, name, iteration, accs[-1, 0, iteration]), savePdf=savePdf)

        np.savez("", all_data)
    # labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    # plot_errs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations",
    #           "_SNR_ITER_{1}".format(SNR, name), savePdf=savePdf)

    return np.mean(EERs, axis=0).T, thresholds


def process_eer(loss_tensor, iterations=10, savePdf=True, SNR=1, name=""):
    all_data = []
    batch_size_array = np.arange(1, 20)
    N_Signals = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = batch_size_array * 0.33
    for iteration in range(iterations):
        for batch_size in batch_size_array:
            temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
            if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
                break

            roc1, roc2, scores, eer_min = calculate_roc(temp_loss_tensor, step=0.01)
            EERs[:, batch_size - 1, iteration] = eer_min[0, :]

            # plot_roc(roc1, roc2, eer_min)
            print("EER MIN: ,{0}, n windows: {1}".format(eer_min[0], np.shape(temp_loss_tensor)[2]))

            all_data.append([iteration, batch_size, eer_min[0, :], scores, roc1, roc2])

        labels = ["ECG {0}".format(i) for i in range(1, N_Signals + 1)]

        plot_errs(EERs.T, seconds, labels, VALIDATION_DIRECTORY, "EER_SNR_{0}{1}_{2}".format(SNR, name, iteration),
                "Mean EER for SNR of {0}".format(SNR), savePdf=savePdf)

    np.savez("../data/validation/ALL_DATA_SNR_{0}_{1}.npz".format(SNR, name), all_data=all_data)
    # labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    # plot_errs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_SNR_ITER_{0}{1}".format(SNR, name), savePdf=savePdf)

    return np.mean(EERs, axis=0).T


# full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,21)))
# processed_noise_array, SNRs = make_noise_signals(full_paths, MIN_NOISE_DB)
# np.savez(noise_filename, SNRs=SNRs, processed_noise_array=processed_noise_array)

def process_all_eers(loss_quaternion, SNRs, iterations=1, loss_iteration=0, batch_size=120):
    mean_EERs = []
    for SNR, loss_tensor in zip(SNRs, loss_quaternion):
        EERs = process_alternate_eer(loss_tensor, iterations=iterations,
                                     savePdf=False, SNR=SNR, name=str(loss_iteration), batch_size=batch_size)
        mean_EERs.append(EERs)

    np.savez(VALIDATION_DIRECTORY + "/eers.npz", EERs=mean_EERs)

    return mean_EERs


def load_tests_cybhi(long=True):
    if long:
        processed_data_path = '../data/biometry_cybhi[256].npz'

    file = np.load(processed_data_path)
    train_dates, train_names, test_dates, test_names, test_signals = \
        file["train_dates"], file["train_names"], file["test_dates"], file["test_names"], file["test_signals"]

    indexes_ = sorted(range(len(test_signals)), key=lambda k: test_names[k])
    train_dates, train_names = train_dates[indexes_], train_names[indexes_]

    indexes = []
    for name in train_names:
        indexes.append(test_names.tolist().index(name))

    return test_signals[indexes]


def load_train_cybhi():
    processed_data_path = '../data/biometry_cybhi[256].npz'

    file = np.load(processed_data_path)
    train_dates, train_names, test_dates, test_names, train_signals = \
        file["train_dates"], file["train_names"], file["test_dates"], file["test_names"], file["train_signals"]

    indexes_ = sorted(range(len(test_signals)), key=lambda k: test_names[k])

    return train_signals[indexes_]


# def prepare_data(windows, signal2model, overlap=0.11, batch_percentage=1):
#     x__matrix, y__matrix = [], []
#     window_size, max_batch_size, mini_batch_size = \
#         signal2model.window_size, signal2model.batch_size, signal2model.mini_batch_size
#     reject = 0
#     total = 0
#     for w, window in enumerate(windows):
#         if len(window) > window_size+1:
#             for s_w in range(0, len(window) - window_size - 1, int(window_size*overlap)):
#                 small_window = np.round(window[s_w:s_w + window_size])
#                 # print(np.max(small_window) - np.min(small_window))
#                 if (np.max(small_window) - np.min(small_window)) \
#                         > 0.9*signal2model.signal_dim:
#                     x__matrix.append(window[s_w:s_w + window_size])
#                     y__matrix.append(window[s_w + 1:s_w + window_size + 1])
#                 else:
#                     reject += 1
#
#                 total += 1
#
#     x__matrix = np.array(x__matrix)
#     y__matrix = np.array(y__matrix)
#     batch_size = max_batch_size if max_batch_size < np.shape(x__matrix)[0] else \
#         np.shape(x__matrix)[0] - np.shape(x__matrix)[0] % mini_batch_size
#     indexes = int(batch_size*batch_percentage) + np.random.permutation(np.shape(x__matrix)[0] - int(batch_size*batch_percentage))
#
#     print("Windows of {0}: {1}; Rejected: {2} of {3}".format(np.shape(indexes)[0], batch_size, reject, total))
#     return x__matrix[indexes], y__matrix[indexes]

if __name__ == "__main__":
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 16
    batch_size = 128
    fs = 250
    window_size = 512
    W = 512
    save_interval = 1000
    iterations = 1
    bs = 60
    all_EERs = []
    seconds = (W / fs) + (np.arange(1, bs) * W * 0.11) / fs
    loss_tensor = []
    mean_EERs = []
    limit = 0
    # prefix = "_LONG_{0}".format(limit)
    prefix = "_LONG_{0}".format(limit)

    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format('NEW', window_size)
    fileDir = "Data/CYBHi"
    signals = np.load(fileDir + "/signals.npz")["signals"]
    signals = signals[list(range(12))+list(range(13,len(signals)))]
    size = np.zeros(len(signals))
    all_models_info = []
    test_signals = []
    full_path = VALIDATION_DIRECTORY + "/LOSS_CYBHi{1}[64.{0}]".format(window_size, prefix)


    for s, signal_data in enumerate(signals):
        name = 'ecg_cybhi_' + signal_data.name
        all_models_info.append(ModelInfo(Sd=signal_dim, Hd=hidden_dim, dataset_name=name, directory=signal_directory,
                                         DS=-5, t=-5, W=W, name="CYBHi {0}".format(s)))
        test_signals.append(signal_data.processed_test_windows)

    RLTC.get_or_save_loss_tensor(full_path, None, W=window_size, models=all_models_info,
                                 test_signals=test_signals, force_new=False, mean_tol=0.8,
                                 overlap=0.11, batch_percentage=0, mini_batch=32, std_tol=1000)

    for i in [15, 40, 58]:
        RLTC.identify_biosignals(loss_tensor, all_models_info, batch_size=i)

    EERs, thresholds = RLTC.process_eers(loss_tensor, W, VALIDATION_DIRECTORY, True, batch_size=60, decimals=5)
    # good_indices = np.where(size > limit)[0]

    # test_signals = [test_signals[i] for i in good_indices]
    # all_models_info = [all_models_info[i] for i in good_indices]
    # n_windows = int(np.min(size[good_indices]))

    # for t, test_signal in enumerate(test_signals):
    #     test_signals[t] = test_signal[:n_windows]
    #
    # print("REJECTED: {0} of {1}".format(len(size) - len(test_signals), len(size)))
    # print("MAX OF WINDOWS: {0}".format(int(np.min(size[good_indices]))))
    # test_signals = np.array(test_signals)
    #
    #
    # # loss_tensor = calculate_loss_tensor(filename, n_windows, W, all_models_info, test_signals)
    # print(filename + " processed")
    #
    #
    # if len(loss_tensor) < 1:
    #     loss_tensor = np.load(filename + ".npz")["loss_tensor"]

    # print("Windows: {0}".format(np.shape(loss_tensor)[2]))
    # x = np.arange(np.shape(loss_tensor)[0]).tolist()
    # # z = 29
    # # x = np.arange(z).tolist() + np.arange(z+1, np.shape(loss_tensor)[0]).tolist()
    # # for i in range(np.shape(loss_tensor)[0]):
    # #     for j in range(np.shape(loss_tensor)[1]):
    # #         for k in range(np.shape(loss_tensor)[2]):
    # #             loss_tensor[i, j, k] = loss_tensor[i, j, k] - np.min(loss_tensor[i, :, :])
    # #             loss_tensor[i, j, k] = loss_tensor[i, j, k] / np.max(loss_tensor[i, :, :])
    # for i in [15, 40, 58]:
    #     classify_biosignals(filename, N_Windows=None, w_for_classification=i, title="")
    #
    # labels = [model.name for model in all_models_info]
    # EERs, thresholds = process_alternate_eer(loss_tensor, iterations=iterations,
    #                              savePdf=True, name="cybhi{0}".format(prefix), batch_size=bs, labels=labels)
