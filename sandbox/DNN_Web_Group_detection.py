import numpy as np

import DeepLibphys.models.libphys_MBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import randomize_batch, plot_confusion_matrix, process_signal, \
    segment_matrix
from DeepLibphys.utils.functions.database import ModelInfo, SignalInfo
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import time
import seaborn

GRU_DATA_DIRECTORY = "../data/trained/"


def load_test_data(filetag=None, dir_name=None):
    print("Loading test data...")
    filename = GRU_DATA_DIRECTORY + dir_name + '/' + filetag + '_test_data.npz'
    npzfile = np.load(filename)
    return npzfile["test_data"]


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


def calculate_loss_tensor(filename, Total_Windows, W, signals_models):
    n_windows = Total_Windows
    if Total_Windows / 256 > 1:
        ratio = round(Total_Windows / 256)
        n_windows = int(Total_Windows / ratio)

    windows = np.arange(int(Total_Windows / n_windows))
    N_Windows = len(windows)
    N_Signals = len(signals_models)
    Total_Windows = int(N_Windows * n_windows)

    loss_tensor = np.zeros((N_Signals, N_Signals, Total_Windows))
    N_Signals = len(signals_models)

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
                x_test = X_matrix[s, index:index + n_windows, :]
                y_test = Y_matrix[s, index:index + n_windows, :]
                loss_tensor[m, s, index:index + n_windows] = np.asarray(model.calculate_loss_vector(x_test, y_test))

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


def get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests, N_Windows, norm=False, min_windows=1):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i].name for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i].name for i in Sig]

    dimensions = 1
    if len(np.shape(loss_tensor)) > 3:
        dimensions = np.size(loss_tensor, axis=3)

    for i in Sig:
        # if signals_models[i].name is not "None":
        loss_tensor[:-1, i, :] = loss_tensor[:-1, i, :] #/ np.max(loss_tensor[:-1, i, :])
    new_min_loss_tensor = np.zeros((np.shape(loss_tensor)[0],
                                   np.shape(loss_tensor)[1],
                                   int(np.shape(loss_tensor)[2]/min_windows),
                                    dimensions))

    # loss_tensor = loss_tensor - np.min(loss_tensor, axis=0)
    # loss_tensor = loss_tensor / np.max(loss_tensor, axis=0)
    index = -1
    for i in range(0, np.shape(loss_tensor)[2]-min_windows+1, min_windows):
        index += 1
        new_min_loss_tensor[:, :, index,:] = np.min(loss_tensor[:,:, i:i+min_windows], axis=2)

    predicted_matrix = []
    if dimensions > 1:
        min_loss_tensor = np.mean(new_min_loss_tensor[Mod][:, Sig, :], axis=3)
        predicted_matrix = np.argmin(min_loss_tensor, axis=0)
    else:
        predicted_matrix = np.argmin(new_min_loss_tensor[Mod][:, Sig, :], axis=0)
        min_loss_tensor = new_min_loss_tensor



    sinal_predicted_matrix = np.zeros((len(Sig), len(Mod)))

    N_Windows = np.shape(new_min_loss_tensor)[2]
    if norm is False:
        N_Windows = 1
    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j) / N_Windows

    return sinal_predicted_matrix, labels_model, labels_signals


def calculate_prediction_matrix(loss_tensor, threshold=1):
    threshold_layer = np.ones((np.shape(loss_tensor)[1], np.shape(loss_tensor)[2])) * threshold
    normalized_loss_tensor = np.zeros(
        (np.shape(loss_tensor)[0] + 1, np.shape(loss_tensor)[1], np.shape(loss_tensor)[2]))
    predicted_tensor = np.zeros_like(loss_tensor)

    for j in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:-1, j, :] = - np.min(loss_tensor[:, j, :], axis=0)
        normalized_loss_tensor[:-1, j, :] = normalized_loss_tensor[:-1, j, :] / np.max(loss_tensor[:, j, :], axis=0)

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
        # plt.plot(classified_matrix)
        # plt.show()

        confusion_tensor[signal, 0, 0] = len(np.where(classified_matrix[signal, :] == 1)[0])  # TP
        confusion_tensor[signal, 0, 1] = len(np.where(classified_matrix[signal, :] == 0)[0])  # FN

        confusion_tensor[signal, 1, 0] = len(
            np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 1)[0])  # FP
        confusion_tensor[signal, 1, 1] = len(
            np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 0)[0])  # TN

    # print(confusion_tensor[0,:,:])
    return confusion_tensor


def print_confusion(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=False, norm=False):
    print(sinal_predicted_matrix)
    plot_confusion_matrix(sinal_predicted_matrix.T, labels_signals, labels_model, norm=norm)
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
    plot_confusion_matrix(sinal_predicted_matrix, labels_model, labels_signals)  # , cmap=cmap)


def classify_biosignals_quaternion(loss_quaternion, signals_models, signals_tests, threshold=300, models_index=None,
                        signals_index=None, N_Windows=None, norm=False, min_windows=1):
    if models_index is None:
        models_index = list(range(np.shape(loss_quaternion)[0]))
    if signals_index is None:
        signals_index = list(range(np.shape(loss_quaternion)[1]))

    dimensions = np.shape(loss_quaternion)[3]

    signals_models = np.hstack((signals_models, ModelInfo(name="None", dataset_name="none")))
    threshold = threshold * np.max(loss_quaternion)
    loss_quaternion = np.vstack(
        (loss_quaternion, threshold * np.ones((1, np.size(loss_quaternion, axis=1), np.size(loss_quaternion, axis=2), np.size(loss_quaternion, axis=3)))))

    sinal_predicted_matrix, signal_labels, model_labels = get_sinal_predicted_matrix(
        models_index, signals_index, loss_quaternion, signals_models, signals_tests, N_Windows, norm=norm,
        min_windows=min_windows)

    labels1, labels2 = [], []
    if np.shape(sinal_predicted_matrix)[0] > np.shape(sinal_predicted_matrix)[1]:
        labels1, labels2 = model_labels, signal_labels
    else:
        labels2, labels1 = model_labels, signal_labels
        sinal_predicted_matrix = sinal_predicted_matrix.T

    print_confusion(sinal_predicted_matrix, labels1, labels2, no_numbers=True, norm=norm)


def classify_biosignals(loss_tensor, signals_models, signals_tests, threshold=300, models_index=None,
                        signals_index=None):
    if models_index is None:
        models_index = list(range(np.shape(loss_tensor)[0]))
    if signals_index is None:
        signals_index = list(range(np.shape(loss_tensor)[1]))



    signals_models = np.hstack((signals_models, ModelInfo(name="None", dataset_name="none")))
    threshold = threshold * np.max(loss_tensor)
    loss_tensor = np.vstack(
        (loss_tensor, threshold * np.ones((1, np.size(loss_tensor, axis=1), np.size(loss_tensor, axis=2)))))

    sinal_predicted_matrix, signal_labels, model_labels = get_sinal_predicted_matrix(
        models_index, signals_index, loss_tensor, signals_models, signals_tests, N_Windows, no_numbers=True)

    labels1, labels2 = [], []
    if np.shape(sinal_predicted_matrix)[0] > np.shape(sinal_predicted_matrix)[1]:
        labels1, labels2 = model_labels, signal_labels
    else:
        labels2, labels1 = model_labels, signal_labels
        sinal_predicted_matrix = sinal_predicted_matrix.T

    print_confusion(sinal_predicted_matrix, labels1, labels2, no_numbers=True)


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

        scores[i] = {"TP": TP,
                     "TN": TN,
                     "FP": FP,
                     "FN": FN,
                     "TPR": TPR,
                     "TNR": TNR,
                     "PPV": PPV,
                     "NPV": NPV,
                     "FPR": FPR,
                     "FDR": FDR,
                     "FNR": FNR,
                     "ACC": ACC,
                     "F1": F1}

        # print(scores[i]["TPR"])
        # if i ==0:
        #     print(confusion_tensor[i])
        #     print("FPR - "+str(scores[i]["TPR"]))
        #     print("FNR - "+str(scores[i]["FNR"]))

    return scores


def calculate_roc(loss_tensor, step=0.001, last_index=1, first_index=0):
    N_Signals = np.shape(loss_tensor)[0]
    eer = np.zeros((2, N_Signals)) - 1

    x = np.arange(first_index, last_index + step, step)
    roc1 = np.zeros((2, len(x), np.shape(loss_tensor)[0]))
    roc2 = np.zeros((2, len(x), np.shape(loss_tensor)[0]))
    for i in range(len(x)):
        if i % 100 == 0:
            print(".", end="")
        scores = calculate_variables(loss_tensor, x[i])
        for j in range(np.shape(loss_tensor)[0]):
            roc1[0, i, j] = scores[j]["FNR"]
            roc1[1, i, j] = scores[j]["FPR"]

            roc2[0, i, j] = scores[j]["TPR"]
            roc2[1, i, j] = scores[j]["FPR"]
            print("ACC {0} - " + str(scores[0]["ACC"]), end=";")
            print("TP {0} - " + str(scores[0]["TP"]), end=";")
            print("FP {0} - " + str(scores[0]["FP"]), end=";")
            print("FN {0} - " + str(scores[0]["FN"]), end=";")
            print("TN {0} - " + str(scores[0]["TN"]))

    remake = False
    candidate_index = 1
    print("end")
    for j in range(np.shape(loss_tensor)[0]):
        # non_zeros = np.where(roc1[0, :, j] > 0)
        # non_zeros1 = np.where(roc2[0, :, j] > 0)
        # non_zeros = np.unique(np.append(np.squeeze(non_zeros), np.squeeze(non_zeros1)))
        candidate = abs(roc1[0, :, j] - roc1[1, :, j])
        candidate_index = np.argmin(candidate)
        eer[0, j] = roc1[0, candidate_index, j]
        eer[1, j] = roc1[1, candidate_index, j]

        if candidate[candidate_index] != 0:
            eer_j, new_fpr_x, new_fpr_y = find_eeq(roc1, j)
            eer[0, j] = eer_j
            eer[1, j] = eer_j
            print(eer_j)



            # if remake and step > 0.0000001:
            #     plot_roc(roc1, roc2, eer)
            #     print("I have to remake.... taking more time....{0}".format(x[candidate_index]))
            #     maxi = candidate_index+100
            #     mini = candidate_index-100
            #     if maxi > len(x):
            #         maxi = -1
            #     if mini <= 0:
            #         mini = 0

            # roc1, roc2, scores, eer = calculate_roc(loss_tensor, step = step*0.01, last_index=x[maxi], first_index=x[mini])

    return roc1, roc2, scores, eer


def plot_roc(roc1, roc2, eer):
    name = 'gnuplot'
    cmap = plt.get_cmap(name)
    cmap_list = [cmap(i) for i in np.linspace(0, 1, N_Signals)]
    fig_1 = "ROC False Negative Rate/False Positive Rate"
    fig_2 = "ROC True Positive Rate/False Positive Rate"
    for j in range(np.shape(loss_tensor)[0]):
        plt.figure(fig_1)
        plt.scatter(roc1[1, :, j], roc1[0, :, j], marker='.', color=cmap_list[j], label='Signal #{0}'.format(j))
        plt.plot(roc1[1, :, j], roc1[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal #{0}'.format(j))

        # plt.figure(fig_2)
        # plt.scatter(roc2[1, :, j], roc2[0, :, j], marker='.', color=cmap_list[j], label='Signal #{0}'.format(j))
        # plt.plot(roc2[1, :, j], roc2[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal #{0}'.format(j))

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

    # plt.figure(fig_2)
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False Positive Rate")
    # plt.ylim([0, 1])
    # plt.xlim([0, 1])
    # plt.legend()
    # plt.show()


def get_min_max(j, interval, roc):
    """

    :param j:
    :param interval:
    :param roc:
    :return:
    """
    diff_roc = roc[0, :, j] - roc[1, :, j]
    index = np.where(diff_roc < 0)[0][0]
    if index == np.shape(roc)[1] - 1:
        return -1

    min_max = [index - interval, index + interval]

    if min_max[0] < 0:
        min_max[1] = interval
        min_max[0] = 0
    elif min_max[1] >= len(roc[0, :, j]):
        min_max[0] -= (len(roc[0, :, j]) + 1 + interval)
        min_max[1] = len(roc[0, :, j]) - 1

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

    x_interpolation = interpolate.splrep(array_indexes, fpr_x, s=0)
    y_interpolation = interpolate.splrep(array_indexes, fnr_y, s=0)

    candidate = np.array([1])
    candidate_index = np.argmin(candidate)
    array_size = 50
    new_fnr_y = []
    while (candidate[candidate_index] > 0.00001):
        new_array_indexes = np.linspace(array_indexes[0], array_indexes[-1], num=array_size)
        try:
            new_fpr_x = interpolate.splev(new_array_indexes, x_interpolation, der=0)
            new_fnr_y = interpolate.splev(new_array_indexes, y_interpolation, der=0)

            # print(new_fnr_y)
            candidate = abs(new_fpr_x - new_fnr_y)
            candidate_index = np.argmin(candidate)
            # print(new_fnr_y[candidate_index], end="..")
            # if len(new_array_indexes) != len(array_indexes):
            # plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
            # plt.scatter(fpr_x, fnr_y, marker='.')
            # plt.show()
            array_size *= 2
        except:
            plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
            plt.scatter(fpr_x, fnr_y, marker='.')
            plt.show()
            break

    eer = new_fnr_y[candidate_index]
    if math.isnan(eer):
        print(new_fpr_x)
        print(new_fnr_y)
    # print(eer)
    return eer, new_fpr_x, new_fnr_y


# CONFUSION_TENSOR_[W,Z]
N_Windows = 400
W = 128

print("Processing Biometric ECG - with #windows of " + str(N_Windows))
filename = "../data/validation/web_loss_quaternion_128_128_[" + str(N_Windows) + "," + str(W) + "]"

web_models_x = db.web_group_x_models_128
web_models_y = db.web_group_y_models_128

npzfile = np.load('web_monitoring_info_trained_128.npz')
web_monitoring_info, new_web_monitoring_info, max_groups, training_indexes, testing_indexes = \
    npzfile["web_monitoring_info"], npzfile["new_web_monitoring_info"], npzfile["max_groups"], \
    npzfile["training_indexes"], npzfile["testing_indexes"]

print("Loading model...")
model_info = web_models_x[0]


signals_info = []
N_people = len(new_web_monitoring_info)
N_models = len(web_models_x)
N_dimensions = 2
N_Windows = 256
mini_batch_size = int(N_Windows/4)

Signal_windows = np.zeros((N_people, 2, N_dimensions, N_Windows, W))

# Signal_windows    -> dim == 0: Number of signals (in this case the number of people)
#                   -> dim == 1: Input and labels
#                   -> dim == 2: Number of dimensions (in this case 2)
#                   -> dim == 3: Number of windows
#                   -> dim == 4: Number of samples in each window
p = -1
signal_labels = []
max = []
for monitoring_info in new_web_monitoring_info:
    try:
        print("length ", len(monitoring_info[1][0])," vs ", N_Windows * W * 3)

        if len(monitoring_info[1][0]) >= (N_Windows * W * 0.33):
            p += 1
            max.append(monitoring_info[3])
            signal_labels.append(monitoring_info[3]) # append maximizer score information
            X_tensor, Y_tensor = segment_matrix(np.asarray([process_signal(monitoring_info[1][0], model_info.Sd, 10, False, None, False),
                                                            process_signal(monitoring_info[2][0], model_info.Sd, 10, False, None, False)]),
                                                W, overlap=0.33)

            X_quaternion, Y_quaternion = randomize_batch(X_tensor, Y_tensor, N_Windows)
            Signal_windows[p, :, :, :, :] = np.asarray([X_quaternion, Y_quaternion])
            print("Calculated for p = "+str(p))
        else:
            print("Not calculated for p = "+str(p))
    except:
        print("Exception!")
        pass

Signal_windows = Signal_windows[~(Signal_windows[:,0,0,0,0] == 0),:,:,:,:]

N_people, inputs, N_dimensions, N_Windows, W = np.shape(Signal_windows)
print(N_dimensions)
loss_quartenion = np.ones((N_models, N_people, N_Windows, N_dimensions))*100000


print("done")
models = np.shape(loss_quartenion)[0]
signals = np.shape(loss_quartenion)[1]
dimensions = np.shape(loss_quartenion)[3]

# model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
#                         n_windows=mini_batch_size)
# m = -1
# for model_x, model_y in zip(web_models_x, web_models_y):
#     m += 1
#
#     for d in range(N_dimensions):
#         if d == 0:
#             model_info = model_x
#         if d == 1:
#             model_info = model_y
#         s = -1
#         model.signal_name = model_info.dataset_name
#         print(model.signal_name)
#         model.load(filetag=model.get_file_tag(-5,-5), dir_name=model_info.directory)
#         for signal_windows in Signal_windows:
#             s += 1
#             for mini_batch_index in list(range(0, N_Windows, mini_batch_size)):
#                 loss_quartenion[m, s, mini_batch_index: mini_batch_index + mini_batch_size, d] = \
#                     model.calculate_loss_vector(signal_windows[0, d, mini_batch_index: mini_batch_index + mini_batch_size, :],
#                                              signal_windows[1, d, mini_batch_index: mini_batch_index + mini_batch_size, :])
#
# np.savez(filename, loss_quartenion=loss_quartenion)
# print(filename + " has been saved!")
max = np.asarray(max)
sorted_index = np.argsort(max)
max = max[sorted_index]
npzfile = np.load(filename+".npz")
loss_quartenion = npzfile["loss_quartenion"]
# loss_tensor_x = loss_quartenion[:, :, :, 0][:, sorted_index, :]
# loss_tensor_y = loss_quartenion[:, :, :, 1][:, sorted_index, :]

loss_quartenion = loss_quartenion[:, sorted_index, :, :]

signals_info = []
for i in range(len(max)):
    signals_info.append(SignalInfo("web", 'Web', i, -1, "WEB X" + str(max[i])))

for i in range(1,21):
    print(i)
    classify_biosignals_quaternion(loss_quartenion,
                                   web_models_x,
                                   signals_info,
                                   threshold=0.5,
                                   N_Windows=N_Windows,
                                   norm=True,
                                   min_windows=i)

# signals_info = []
# for i in range(len(max)):
#     signals_info.append(SignalInfo("web", 'Web', i, -1, "WEB Y" + str(max[i])))
#
# classify_biosignals(loss_tensor_y, web_models_y, signals_info, threshold=0.5)

# for iteration in range(10):
#
#
#     # fine_loss_tensor = calculate_fine_loss_tensor(filename, N_Windows, W, signals_models)
#     #
#     # npzfile = np.load(filename + ".npz")
#     # fine_loss_tensor, signals_models,  = \
#     #     npzfile["loss_tensor"], npzfile["signals_models"]
#
#     N_Signals = np.shape(loss_tensor)[0]
#     N_Total_Windows = np.shape(loss_tensor)[2]
#
#     eers = [[],[]]
#
#     batch_size_array = np.arange(1, 241)
#     for batch_size in batch_size_array:
#         batch_indexes = np.arange(0, N_Total_Windows-batch_size, batch_size)
#         N_Batches = len(batch_indexes)
#         Total_Windows = N_Batches*batch_size
#         if N_Batches == 0:
#             break
#
#         temp_loss_tensor = np.zeros((N_Signals, N_Signals, N_Batches))
#         print(batch_size, end=" - ")
#         print(N_Batches)
#
#         loss_tensor = loss_tensor[:,:,np.random.permutation(N_Total_Windows)]
#         x = 0
#         for i in batch_indexes:
#             loss_batch = loss_tensor[:, :, i:i+batch_size]
#             temp_loss_tensor[:, :, x] = np.min(loss_batch, axis=2)
#             x += 1
#
#
#
#         roc1, roc2, scores, eer_min = calculate_roc(temp_loss_tensor, step=0.001)
#         plot_roc(roc1, roc2, eer_min)
#         print("EER MIN: {0}".format(eer_min[0]))
#         eers[0].append(eer_min[0, :])
#         eers[1].append(eer_min[1, :])
#
#
#     np.savez(filename + "_eer_"+str(iteration)+".npz",
#              eers=eers,
#              mean_eer=np.mean(eers[0],axis=1))
#
#     plt.figure(iteration)
#     seconds = batch_size_array[:len(eers[0])]*0.33
#
#     plt.plot(seconds, np.mean(eers[0],axis=1), 'b.', label="EER MEAN")
#     plt.plot(seconds, np.mean(eers[0],axis=1), alpha=0.5, label="EER MEAN")
#     plt.plot(seconds, eers[0], alpha=0.2)
#     index_min = np.argmin(np.mean(eers[0],axis=1))
#     plt.plot(seconds[index_min], np.mean(eers[0],axis=1)[index_min],'ro', alpha=0.6)
#     plt.annotate("EER MIN = {0:5}".format(np.mean(eers[0],axis=1)[index_min]), xy=(seconds[index_min], np.mean(eers[0],axis=1)[index_min]+0.005))
#     plt.xlabel("Seconds of signal")
#     plt.ylabel("Equal Error Rate")
#     # for i in range(len(eers[0])):
#     #     plt.figure()
#     #     plt.plot(eers[0][i], alpha=0.2)
#     #     plt.plot(np.mean([eers[0,i,:],eers[1][i,:]], axis=0))
#     #     plt.plot(eers[0][i], alpha=0.2)
#     # index_min = np.argmin(np.array(eers)[np.array(eers)>0])
#     # plt.plot(batch_size_array[index_min], eers[index_min],)
#     # plt.annotate("EER MIN = {0}".format(eers[index_min]), xy=(batch_size_array[index_min], eers[index_min]))
#     plt.legend()
#     plt.show()
