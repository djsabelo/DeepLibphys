import numpy as np

import DeepLibphys.models.libphys_MBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.database import ModelInfo
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import math
import time
import seaborn
from matplotlib.backends.backend_pdf import PdfPages

GRU_DATA_DIRECTORY = "../data/trained/"

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

def calculate_loss_tensor(filename, Total_Windows, W, signals_models, signals=None, noisy_index=None):

    n_windows = Total_Windows
    if Total_Windows / 256 > 1:
        ratio = round(Total_Windows / 256)
    else:
        ratio = 1
    n_windows = 250

    windows = np.arange(int(Total_Windows/n_windows))
    N_Windows = len(windows)
    N_Signals = len(signals_models)
    Total_Windows = int(N_Windows*n_windows)

    loss_tensor = np.zeros((N_Signals, N_Signals, Total_Windows))
    N_Signals = len(signals_models)

    X_matrix = np.zeros((N_Signals, Total_Windows, W))
    Y_matrix = np.zeros((N_Signals, Total_Windows, W))

    i = 0
    indexes = signals_models#[np.random.permutation(len(signals_models))]
    for model_info in indexes:
        if signals is None:
            # [x_test, y_test] = load_test_data("GRU_" + model_info.dataset_name, + "["+str(model_info.Sd)+"."+str(model_info.Hd)+".-1.-1.-1]"
            #                               , model_info.directory)
            [x_test, y_test] = load_test_data(model_info.dataset_name, model_info.directory)
            X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(x_test, y_test, Total_Windows)
        else:
            signals = get_signals_tests(db.ecg_noisy_signals[noisy_index-1], index=i, noisy_index=noisy_index,
                                        peak_into_data=False)
            signal_test = segment_signal(signals[0][i], 256, 0.33)
            X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(signal_test[0], signal_test[1], Total_Windows)

        i += 1

    print("Loading model...")
    model_info = signals_models[0]

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
                loss_tensor[m, s, index:index+n_windows] = np.asarray(model.calculate_loss_vector(x_test, y_test))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models_=indexes,
             signals_models=signals_models)

    return loss_tensor, indexes


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

    x = np.arange(first_index, last_index + step, step)
    roc1 = np.zeros((2, len(x), np.shape(loss_tensor)[0]))
    roc2 = np.zeros((2, len(x), np.shape(loss_tensor)[0]))
    for i in range(len(x)):
        if i%100==0:
            print(".", end="")
        scores = calculate_variables(loss_tensor, x[i])
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
            print(eer_j)



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


    return roc1, roc2, scores, eer


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

    mean_EERs = np.mean(EERs, axis=0)
    index_min = np.argmin(mean_EERs)
    plt.plot(time, EER, 'b.', alpha=0.5)
    plt.plot(time, EER, 'b-', alpha=0.5, label="Mean")
    plt.plot(time[index_min], mean_EERs[index_min], 'ro', alpha=0.6)
    indexi = 0.6 * np.max(time)
    indexj = 0.5 * (np.max(EER) - np.min(EER))

    step = 0.1 * (np.max(EER) - np.min(EER))
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
        pdf = PdfPages("img/EER{0}.pdf".format(file))
        pdf.savefig(fig)
        pdf.close()
    else:
        plt.show()
    return plt

def process_EER(loss_tensor, N_Windows, W, iterations=10):
    batch_size_array = np.arange(1, 60)
    N_Signals = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = batch_size_array * 0.33
    for iteration in range(iterations):

        for batch_size in batch_size_array:
            temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
            if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
                break;

            roc1, roc2, scores, eer_min = calculate_roc(temp_loss_tensor, step=0.001)
            EERs[:, batch_size-1, iteration] = eer_min[0, :]
            # plot_roc(roc1, roc2, eer_min)
            print("EER MIN: ,{0}".format(eer_min[0]))

        labels = ["ECG {0}".format(i) for i in range(1, N_Signals+1)]

        plot_EERs(EERs[:, :, iteration], seconds, labels, "Mean EER", "_X")

    labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    plot_EERs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_X_iter")
    return EERs

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
W = 256
N_Windows = 7000


print("Processing Biometric ECG - with #windows of "+str(N_Windows))
signals_models = db.ecg_clean_models
signals_models = np.array(signals_models)

MIN_NOISE_DB = 17
#
# noise_filename = "../data/ecg_noisy_signals[17].npz"
noise_filename = "../data/ecg_noisy_signals.npz"

# full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,21)))
# processed_noise_array, SNRs = make_noise_signals(full_paths, MIN_NOISE_DB)
# np.savez(noise_filename, SNRs=SNRs, processed_noise_array=processed_noise_array)

npzfile = np.load(noise_filename)
processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]
# CONFUSION_TENSOR_[W,Z]
N_Windows = 7000
W = 256
signals_models = db.ecg_clean_models


loss_tensors = []
titles = []

# [SNR, person, sample]

# for model in signals_models:
# filename = '../data/validation/NOISY_[17]_INDEX_7000'
filename = '../data/validation/NOISY_INDEX_7000'

signals_xxx = []
for signals, SNR in zip(processed_noise_array, SNRs):
    for signal in signals:
        signals_xxx.append(signal)
processed_noise_array = []

# loss_tensor = calculate_loss_tensor(filename, N_Windows, W, signals_xxx, signals_models)
m_labels = [model_info.name for model_info in signals_models]

loss_tensors = np.load(filename+".npz")['loss_tensor']
loss_quartenion = np.zeros((len(SNRs), np.shape(loss_tensors)[0], np.shape(loss_tensors)[0], N_Windows))
step = np.shape(loss_tensors)[0]
titles = []
for i, SNR in zip(range(len(SNRs)), SNRs):
    title = "SIGNAL OF SNR of " + str(SNR)
    print("Processing "+title)
    loss_quartenion[i] = loss_tensors[:, i*step:i*step+step, :]
    titles.append(title)

s_labels = ["ECG {0}".format(i) for i in range(1, np.shape(loss_quartenion)[1]+1)]
i = 0
for loss_tensor in loss_quartenion:

    # classified_matrix = calculate_classification_matrix(loss_tensor)
    process_EER(loss_tensor, N_Windows, W, iterations=10)
    # process_EER(loss_tensor, N_Windows, W, 256, titles[i], s_labels)
    # print_confusion(classified_matrix, s_labels, m_labels)
    i+=1



