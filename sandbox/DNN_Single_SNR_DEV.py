import numpy as np

import DeepLibphys
import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.database import ModelInfo
from DeepLibphys.utils.functions.signal2model import Signal2Model
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import math
from itertools import repeat
import time
import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

GRU_DATA_DIRECTORY = "../data/trained/"
SNR_DIRECTORY = "../data/validation/June_DNN_SNR_FANTASIA_1024_256"


def load_test_data(filetag=None, dir_name=None):
    print("Loading test data...")
    filename = GRU_DATA_DIRECTORY + dir_name + '/' + filetag + '_test_data.npz'
    npzfile = np.load(filename)
    return npzfile["test_data"]


def make_noise_signals(full_paths, max_target_SNR=16):
    target_SNR_array = np.arange(max_target_SNR, max_target_SNR-5, -1)
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

    while int(SNR*100) != target_SNR*100:

        added_noise = np.random.normal(0, last_std, len(signal))
        signal_with_noise = np.array(np.array(signal) + added_noise)
        SNR = calculate_signal_to_noise_ratio(signal_with_noise, smoothed_signal)

        if int(SNR*100) > target_SNR*100:
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
    n_windows = 256

    windows = np.arange(0, Total_Windows - n_windows+1, n_windows)
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
        signal_test = segment_signal(signal[first_test_index:], W, 0.33)
        X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(signal_test[0], signal_test[1], Total_Windows)

        i += 1

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
                                             repeat(model), repeat(X_matrix), repeat(Y_matrix), repeat(n_windows), repeat(windows), repeat(N_Signals),
                                             repeat(loss_tensor)))

    if not os.path.isdir(os.path.dirname(filename + ".npz")):
        os.mkdir(os.path.dirname(filename + ".npz"))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models)

    return loss_tensor


def calculate_fine_loss_tensor(filename, Total_Windows, W, signals_models, n_windows):
    windows = np.arange(int(Total_Windows/n_windows))
    N_Windows = len(windows)
    N_Signals = len(signals_models)
    Total_Windows = int(N_Windows*n_windows)

    loss_tensor = np.zeros((N_Signals, N_Signals, N_Windows))

    X_matrix = np.zeros((N_Signals, Total_Windows, W))
    Y_matrix = np.zeros((N_Signals, Total_Windows, W))

    i = 0
    for model_info in signals_models:
        [x_test, y_test] = load_test_data("GRU_" + model_info.dataset_name + "["+str(model_info.Sd)+"."+str(model_info.Hd)+".-1.-1.-1]"
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
                x_test = X_matrix[s, index:index+n_windows, :]
                y_test = Y_matrix[s, index:index+n_windows, :]
                loss_tensor[m, s, w] = np.asarray(model.calculate_loss(x_test, y_test))

    np.savez(filename + "_fine.npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models)

    return loss_tensor


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


def get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests, N_Windows, no_numbers=False):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i].name for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i].name for i in Sig]

    return calculate_classification_matrix(loss_tensor[Mod][:, Sig, :]), labels_model, labels_signals


def calculate_classification_matrix(loss_tensor):
    N_Windows = np.shape(loss_tensor)[2]
    normalized_loss_tensor = np.zeros_like(loss_tensor)
    for i in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:, i, :] = loss_tensor[:, i, :] - np.min(loss_tensor[:, i, :], axis=0)
        normalized_loss_tensor[:, i, :] = normalized_loss_tensor[:, i, :] / np.max(normalized_loss_tensor[:, i, :], axis=0)

    predicted_matrix = np.argmin(normalized_loss_tensor, axis=0)

    sinal_predicted_matrix = np.zeros((np.shape(loss_tensor)[0], np.shape(normalized_loss_tensor)[1]))

    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j) / N_Windows

    return sinal_predicted_matrix


def calculate_prediction_matrix(loss_tensor, threshold = 1):
    threshold_layer = np.ones((np.shape(loss_tensor)[1],np.shape(loss_tensor)[2])) * threshold
    normalized_loss_tensor = np.zeros((np.shape(loss_tensor)[0]+1,np.shape(loss_tensor)[1],np.shape(loss_tensor)[2]))
    predicted_tensor = np.zeros_like(loss_tensor)

    for j in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:-1, j, :] = loss_tensor[:, j, :] - np.min(loss_tensor[:, j, :], axis=0)
        normalized_loss_tensor[:-1, j, :] = normalized_loss_tensor[:-1, j, :] / np.max(loss_tensor[:, j, :], axis=0)

    # normalized_loss_tensor[:-1,:,:] = loss_tensor
    normalized_loss_tensor[-1, :, :] = threshold_layer
    # print("\nThreshold "+str(threshold), end=":")

    # plt.figure("LOSS NORMALIZED")
    # plt.plot(normalized_loss_tensor[:, 1, 0])
    # plt.plot(normalized_loss_tensor[:, 1, 0])
    # plt.show()

    for i in range(np.shape(loss_tensor)[1]):
        predicted_tensor[i, : , :] = np.argmin(normalized_loss_tensor[[-1, i], :, :], axis=0)

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
        confusion_tensor[signal, 0, 0] = len(np.where(signal_predicted_matrix[signal, :] == signal)[0]) # TP
        confusion_tensor[signal, 0, 1] = len(np.where(signal_predicted_matrix[signal, :] != signal)[0]) # FN

        confusion_tensor[signal, 1, 0] = len(np.where(signal_predicted_matrix[np.arange(np.shape(signal_predicted_matrix)[0]) != signal, :] == signal)[0]) # FP
        confusion_tensor[signal, 1, 1] = len(np.where(signal_predicted_matrix[np.arange(np.shape(signal_predicted_matrix)[0]) != signal, :] != signal)[0]) # TN

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


        confusion_tensor[signal, 0, 0] = len(np.where(classified_matrix[signal, :] == 1)[0]) # TP
        confusion_tensor[signal, 0, 1] = len(np.where(classified_matrix[signal, :] == 0)[0]) # FN

        confusion_tensor[signal, 1, 0] = len(np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 1)[0]) # FP
        confusion_tensor[signal, 1, 1] = len(np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 0)[0]) # TN

    # print(confusion_tensor[0,:,:])
    return confusion_tensor


def print_confusion(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=False,norm=True):
    print(sinal_predicted_matrix)
    plot_confusion_matrix(sinal_predicted_matrix, labels_signals, labels_model, no_numbers, norm=norm)
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
        F1 = 2*TP / (2*TP + FP + FN)

        scores[i] = {  "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN":FN,
                    "TPR":TPR,
                    "TNR":TNR,
                    "PPV":PPV,
                    "NPV":NPV,
                    "FPR":FPR,
                    "FDR":FDR,
                    "FNR":FNR,
                    "ACC":ACC,
                    "F1":F1}

        # print(scores[i]["TPR"])
        # if i ==0:
        #     print(confusion_tensor[i])
        #     print("FPR - "+str(scores[i]["TPR"]))
        #     print("FNR - "+str(scores[i]["FNR"]))


    return scores


def calculate_roc(loss_tensor, step = 0.001, last_index=1, first_index=0):
    last_index += step
    first_index -= step
    N_Signals = np.shape(loss_tensor)[0]
    eer = np.zeros((2, N_Signals)) - 1

    thresholds = np.arange(first_index, last_index + step, step)
    n_thresholds = len(thresholds)
    roc1 = np.zeros((2, n_thresholds, np.shape(loss_tensor)[0]))
    roc2 = np.zeros((2, n_thresholds, np.shape(loss_tensor)[0]))
    for i in range(n_thresholds):
        if i % 100 == 0:
            print(".", end="")
        scores = calculate_variables(loss_tensor, thresholds[i])
        for j in range(np.shape(loss_tensor)[0]):
            roc1[0, i, j] = scores[j]["FNR"]
            roc1[1, i, j] = scores[j]["FPR"]

            roc2[0, i, j] = scores[j]["TPR"]
            roc2[1, i, j] = scores[j]["FPR"]
            # print("ACC {0} - "+str(scores[0]["ACC"]), end =";")
            # print("TP {0} - " + str(scores[0]["TP"]), end=";")
            # print("FP {0} - " + str(scores[0]["FP"]), end=";")
            # print("FN {0} - " + str(scores[0]["FN"]), end=";")
            # print("TN {0} - " + str(scores[0]["TN"]))

    remake = False
    candidate_index = 1
    print("end")
    candidate_index = 0
    for j in range(np.shape(loss_tensor)[0]):
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


    return roc1, roc2, scores, eer, thresholds[candidate_index]


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
    if(len(np.where(diff_roc < 0)[0])==0):
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
        min_max[0] -= (len(roc[0, :, j])+1+interval)
        min_max[1] = len(roc[0, :, j])-1
        if min_max[0] < 0:
            min_max[1] = interval
            min_max[0] = 0

    return min_max


def find_eeq(roc, j):
    interval = 10
    min_max = get_min_max(j, interval, roc)
    roc_min_max = roc[:,min_max,j]

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
        print("Patience is exausted when searching for eer, minimum found: "+str(candidate[candidate_index]))
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
        print("Saving img/EER{0}.pdf".format(file))
        pdf = PdfPages("img/EER{0}.pdf".format(file))
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
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = batch_size_array * 0.33
    for iteration in range(iterations):
        for batch_size in batch_size_array:
            temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
            if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
                break

            roc1, roc2, scores, eer_min = calculate_roc(temp_loss_tensor, step=0.001)
            EERs[:, batch_size-1, iteration] = eer_min[0, :]

            # plot_roc(roc1, roc2, eer_min)
            print("EER MIN: ,{0}".format(eer_min[0]))

            all_data.append([iteration, batch_size, eer_min[0, :], scores, roc1, roc2])

        labels = ["ECG {0}".format(i) for i in range(1, N_Signals+1)]

        plot_EERs(EERs[:, :, iteration], seconds, labels, "Mean EER for SNR of {0}".format(SNR),
                  "_SNR_{0}{1}_{2}".format(SNR, name, iteration), savePdf=savePdf)

    np.savez(SNR_DIRECTORY+"/ALL_DATA_SNR_{0}{1}.npz".format(SNR, name), all_data = all_data)
    labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    plot_EERs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_SNR_ITER_{0}{1}".format(SNR, name), savePdf=savePdf)

    return np.mean(EERs, axis=0).T

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
    randomized_loss_tensor = loss_tensor[:, :, np.random.permutation(N_Total_Windows)]
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
        first_index = int(0.33 * len(all_signals[0,0]))
        _, _, N_Windows, _ = segment_signal(all_signals[0,0], W, 0.33, None, first_index)

    for models_info, filename, signals in zip(all_models_info, filenames, all_signals):
        loss_quartenion.append(calculate_loss_tensor(filename, N_Windows, W, models_info, signals))

    return loss_quartenion

def calculate_eers(batch_size, loss_tensor, iteration):
    print("batch_size = {0}, iteration = {1}".format(batch_size, iteration))
    temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
    if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
        return

    roc1, roc2, scores, eer_min, threshold = calculate_roc(temp_loss_tensor, step=0.001)
    # plot_roc(roc1, roc2, eer_min)
    print("EER MIN of batch_size = {0}, iteration = {1}: {2}".format(batch_size, iteration, np.min(eer_min[0])))

    return [eer_min[0, :], iteration, batch_size, scores, roc1, roc2, threshold]

def descompress_data(data):
    EERs, iteration, batch_size, scores, roc1, roc2, thresholds = [], [], [], [], [], []
    for line in data:
        EERs.append(line[0])
        iteration = line[1]
        batch_size.append(line[2])
        scores.append(line[3])
        roc1.append(line[4])
        roc2.append(line[5])
        thresholds.append(line[6])

    EERs = np.array(EERs).T
    new_data = {
        "EERs": EERs,
        "iteration": iteration,
        "batch_size": batch_size,
        "scores": scores,
        "roc1": roc1,
        "roc2": roc2,
        "best": np.argmin(np.mean(EERs, axis=0)),
        "theshold": thresholds
    }
    return EERs, iteration, batch_size, scores, roc1, roc2, new_data, thresholds


def process_alternate_eer(loss_tensor, iterations=10, savePdf=True, SNR=1, name="", batch_size=120, fs = 250):
    all_data = []
    batch_size_array = np.arange(1, batch_size)
    N_Signals = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = batch_size_array * 0.33 * W / fs
    all_data = []
    for iteration in range(iterations):
        pool = Pool()
        x = np.round(time.time() * 10 ** 7) * iteration / np.random.randint(10 ** 7, (10 ** 8 - 1), 1)
        if x < 2**32 - 1:
            np.random.seed(int(x))
        data = pool.starmap(calculate_eers, zip(batch_size_array, repeat(loss_tensor), repeat(iteration)))
        pool.close()
        EER, epoch, batch_size, scores, roc1, roc2, save_data, thresholds = descompress_data(data)
        EERs[:, :, iteration] = np.array(EER)
        all_data.append(data)
        labels = ["ECG {0}".format(i) for i in range(1, N_Signals+1)]

        plot_errs(EERs[:, :, iteration], seconds, labels, "Mean EER for SNR of {0}".format(SNR),"_SNR_{0}_{1}".format(SNR, name, iteration), savePdf=savePdf)

        all_data.append(save_data)
    np.savez(SNR_DIRECTORY+"/ALL_DATA_SNR_{0}_{1}.npz".format(SNR, name), all_data=all_data)
    labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    # plot_errs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_SNR_ITER_{1}".format(SNR, name), savePdf=savePdf)

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
            EERs[:, batch_size-1, iteration] = eer_min[0, :]

            # plot_roc(roc1, roc2, eer_min)
            print("EER MIN: ,{0}".format(eer_min[0]))

            all_data.append([iteration, batch_size, eer_min[0, :], scores, roc1, roc2])

        labels = ["ECG {0}".format(i) for i in range(1, N_Signals+1)]

        plot_errs(EERs.T, seconds, labels, "Mean EER for SNR of {0}".format(SNR),
                  "_SNR_{0}{1}_{2}".format(SNR, name, iteration), savePdf=savePdf)

    np.savez("../data/validation/ALL_DATA_SNR_{0}_{1}.npz".format(SNR, name), all_data = all_data)
    # labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    # plot_errs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_SNR_ITER_{0}{1}".format(SNR, name), savePdf=savePdf)

    return np.mean(EERs, axis=0).T


def plot_errs(EERs, time, labels, title="", file="_iterations", savePdf=False, plot_mean=True):
    cmap = matplotlib.cm.get_cmap('rainbow')
    fig = plt.figure("fig", figsize=(900 / 96, 600 / 96), dpi=96)
    for i, EER in zip(range(len(EERs)), EERs):
        plt.plot(time, EER, '.', color=cmap(i / len(EERs)), alpha=0.2)
        plt.plot(time, EER, color=cmap(i / len(EERs)), alpha=0.2, label=labels[i])

    if plot_mean:
        mean_EERs = np.mean(EERs, axis=0).squeeze()
        index_min = np.argmin(mean_EERs)
        plt.plot(time, mean_EERs, 'b.', alpha=0.5)
        plt.plot(time, mean_EERs, 'b-', alpha=0.5, label="Mean")
        plt.plot(time[index_min], np.min(mean_EERs), 'ro', alpha=0.6)
    else:
        for i, EER in zip(range(len(EERs)), EERs):
            index_min = np.argmin(EER)
            plt.plot(time[index_min], np.min(EER), 'o', color=cmap(i / len(EERs)), alpha=0.6)

    indexi = 0.6 * np.max(time)
    indexj = 0.5 * (np.max(EER) - np.min(EER))

    step = (np.max(EER) - np.min(EER))
    if plot_mean:
        plt.annotate("EER MIN MEAN = {0:.4f}".format(mean_EERs[index_min]),
                     xy=(indexi, indexj))

        # plt.annotate("EER MIN STD = {0:.4f}".format(np.std(EER)),
        #              xy=(indexi, indexj - step))
        #
        # plt.annotate("TIME FOR MIN MEAN = {0:.4f}".format(time[index_min]),
        #              xy=(indexi, indexj - step * 2))
        #
        # plt.annotate("TIME FOR MIN STD = {0:.4f}".format(mean_EERs[index_min]),
        #              xy=(indexi, indexj - step * 3))
    else:
        for i, EER in zip(range(len(EERs)), EERs):
            index_min = np.argmin(EER)
            plt.annotate("EER MIN = {0:.3f}%".format(EER[index_min]*100),
                         xy=(time[index_min], EER[index_min] + 0.01), color=cmap(i / len(EERs)), ha='center')

    plt.legend()
    plt.title(title)

    # if savePdf:
    print("Saving img/EER{0}.pdf".format(file))
    dir_name = SNR_DIRECTORY+"/img"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pdf = PdfPages(dir_name + "/EER{0}.pdf".format(file))
    plt.savefig(dir_name + "/EER{0}.eps", format='eps', dpi=1000)
    pdf.savefig(fig)
    plt.clf()
    pdf.close()
    # else:
    #     plt.show()
    return plt


# full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,21)))
# processed_noise_array, SNRs = make_noise_signals(full_paths, MIN_NOISE_DB)
# np.savez(noise_filename, SNRs=SNRs, processed_noise_array=processed_noise_array)

def process_all_eers(loss_quaternion, SNRs, iterations=1, loss_iteration=0, batch_size=120):
    mean_EERs = []
    for SNR, loss_tensor in zip(SNRs, loss_quaternion):
        EERs, thresholds = process_alternate_eer(loss_tensor, iterations=iterations,
                               savePdf=False, SNR=SNR, name=str(loss_iteration), batch_size=batch_size)
        mean_EERs.append(EERs)

    np.savez("eers.npz", EERs=mean_EERs)

    return mean_EERs

# CONFUSION_TENSOR_[W,Z]

# signals_models = db.ecg_clean_models



#########
# loss_quartenion = [np.load("../data/CLEAN_1024" + ".npz")["loss_tensor"]]
# description = "NOISY_INDEX_WITH_NOISE_"
#
# signals_models = [db.ecg_SNR_12, db.ecg_SNR_11, db.ecg_SNR_9, db.ecg_SNR_8]
#
# SNRs = SNRs[[0, 1, 3, 4]]
# [loss_quartenion.append(np.load('../data/validation/{0}{1}'.format(description, snr) + ".npz")["loss_tensor"]) for
#  snr, signal_models in zip(SNRs, signals_models)]
#
# SNRs = [-1] + SNRs
# labels = [str(SNR) for SNR in SNRs]
# time = np.arange(1, 60) * 0.33
# mean_EERs = np.zeros((len(SNRs), len(time)))
#
# def process_parallel_EERs(n, SNR, loss_tensor):
#     print("SNR of " + str(SNR))
#     return process_eer(loss_tensor, iterations=1, savePdf=False, SNR=SNR, name=description)
#
# pool = Pool()
# zz = pool.starmap(process_parallel_EERs, zip(range(len(SNRs)), SNRs, loss_quartenion))
#
# print(zz)

if __name__ == "__main__":
    N_Windows = None
    W = 1024
    signal_dim = 256
    hidden_dim = 256
    batch_size = 128
    window_size = 1024
    fs = 250
    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size, signal_dim)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory

    # Noise signals:
    # noise_filename = "../data/ecg_noisy_signals.npz"
    noise_filename = dir_name + "/signals_without_noise_[{}].npz".format(signal_dim)
    npzfile = np.load(noise_filename)
    processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]
    processed_noise_array = processed_noise_array[np.where(np.logical_or(SNRs>9,SNRs==7))[0]]

    raw_filename = dir_name + "/processed_raw_signals_[{}].npz".format(signal_dim)
    # processed_clean_array, y_train = get_fantasia_dataset(signal_dim, range(1, 21)) #np.load("../data/signals_without_noise.npz")['signals_without_noise']
    # np.savez(raw_filename, processed_clean_array=processed_clean_array)
    processed_clean_array = np.load(raw_filename)["processed_clean_array"]

    all_models_info = [db.ecg_1024_256_RAW, db.ecg_1024_256_SNR_12, db.ecg_1024_256_SNR_11, db.ecg_1024_256_SNR_10, db.ecg_1024_256_SNR_7]#, db.ecg_SNR_9, db.ecg_SNR_8]
    SNRs = ["RAW", "12", "11", "10", "7"]#, "9", "8"]


    loss_quaternion = []
    all_signals = np.vstack((
        np.reshape(processed_clean_array, (1,np.shape(processed_clean_array)[0], np.shape(processed_clean_array)[1])),
                   processed_noise_array))

    iterations = 1
    bs = 60
    all_EERs = []
    seconds = (np.arange(1, bs) * W * 0.33) / fs
    indexes = list(range(1,6))+list(range(7,20))
    for iteration in range(iterations):
        filenames = [SNR_DIRECTORY + "/LOSS_FOR_SNR_{0}_iteration_{1}".format(SNR, iteration) for SNR in SNRs]
        loss_quaternion = calculate_all_loss_tensors(all_models_info, all_signals, filenames, N_Windows=N_Windows, W=W)

        if len(loss_quaternion) < 1:
            for filename in filenames:
                loss_tensor = np.load(filename + ".npz")["loss_tensor"]
                loss_quaternion.append(loss_tensor)
            # np.savez(filename, loss_tensor=npzfile["loss_tensor"])

        EERs = process_all_eers(loss_quaternion[:, indexes, indexes, :], SNRs, iterations=1, loss_iteration=iteration, batch_size=bs)
        EERs = np.load("eers.npz")['EERs']
        # EERs = np.reshape(EERs, (np.shape(EERs)[0], np.shape(EERs)[2]))
        # EERs = np.mean(EERs, axis=0)
        if len(all_EERs) == 0:
            all_EERs = EERs
        else:
            all_EERs = np.vstack((all_EERs, EERs))
        # plot_errs(EERs, seconds, SNRs, "Mean EER for ALL SNR", "SNR_ITER_{0}".format(iteration), savePdf=True, plot_mean=True)

    # np.savez(SNR_DIRECTORY + "all_EERs.npz", all_EERs=all_EERs)
    all_data = []
    EERs = np.zeros((len(SNRs), iterations, len(seconds)))
    accurancies = []
    all_ROCs = []
    for snr, SNR in zip(range(len(SNRs)), SNRs):
        ROCs = [[], []]
        accurancy = []
        for iteration in range(iterations):
            filename = SNR_DIRECTORY + "/ALL_DATA_SNR_{0}_{1}".format(SNR, iteration)
            all_data = np.load(filename + ".npz")["all_data"][0]
            _, _, _, _, _, _, new_data = descompress_data(all_data)
            EERs[snr, iteration, :] = np.mean(new_data["EERs"], axis=0)
            ROCs[0].append(new_data["roc1"])
            ROCs[1].append(new_data["roc2"])
            accurancies.append(new_data["scores"])

        all_ROCs.append(ROCs)


    np.savez(SNR_DIRECTORY + "all_EERs.npz", EERs=EERs)
    # EERs = np.load(SNR_DIRECTORY + "all_EERs.npz")["EERs"]
    plot_errs(np.mean(EERs, axis=1), seconds, SNRs, "Mean EER for ALL SNR", "all_SNR", savePdf=True, plot_mean=False)