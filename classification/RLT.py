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
from matplotlib.backends.backend_pdf import PdfPages


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


def calculate_loss_tensor(signals_models_info, X_test_matrix, Y_test_matrix, method="mse"):
    print("Loading model...")
    model_info = signals_models_info[0]
    n_models = np.shape(signals_models_info)[0]
    n_signals = np.shape(X_test_matrix)[0]
    n_classification_windows = np.shape(X_test_matrix)[1]
    window_batch = n_classification_windows
    windows_indexes = [n_classification_windows]

    if int(n_classification_windows / 500) > 1:
        ratio = int(n_classification_windows / 500)
        window_batch = int(n_classification_windows/ratio)

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
                if method == 'mse':
                    loss_tensor[m, s, w:w + window_batch] = np.asarray(
                        model.calculate_mse_vector(
                            X_test_matrix[s, w:w + window_batch, :],
                            Y_test_matrix[s, w:w + window_batch, :]))
                else:
                    loss_tensor[m, s, w:w+window_batch] = np.asarray(
                        model.calculate_loss_vector(
                            X_test_matrix[s, w:w + window_batch, :],
                            Y_test_matrix[s, w:w + window_batch, :]))


def calculate_classification_matrix(loss_tensor):
    n_windows = np.shape(loss_tensor)[2]
    normalized_loss_tensor = normalize_by_max_loss(loss_tensor)

    predicted_matrix = np.argmin(normalized_loss_tensor, axis=0)

    sinal_predicted_matrix = np.zeros((np.shape(loss_tensor)[0], np.shape(normalized_loss_tensor)[1]))

    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j) / n_windows

    return sinal_predicted_matrix


def normalize_by_max_loss(loss_tensor):
    normalized_loss_tensor = np.zeros_like(loss_tensor)
    for i in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:, i, :] = loss_tensor[:, i, :] - np.min(loss_tensor[:, i, :], axis=0)
        normalized_loss_tensor[:, i, :] = normalized_loss_tensor[:, i, :] / np.max(normalized_loss_tensor[:, i, :],
                                                                                   axis=0)

    return normalized_loss_tensor


def normalize_by_probability(loss_tensor):
    normalized_loss_tensor = np.zeros_like(loss_tensor)
    for i in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:, i, :] = 1 - (loss_tensor[:, i, :] / np.sum(
            normalized_loss_tensor[:, i, :],
            axis=0))

    return normalized_loss_tensor


def calculate_eer(loss_tensor, max_n_windows=120, plot_roc=False):
    batch_size_array = np.arange(1, max_n_windows)
    n_signals = np.shape(loss_tensor)[1]
    EERs = np.zeros((n_signals, max_n_windows))
    for batch_size in batch_size_array:
        temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
        if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
            break;

        roc1, roc2, scores, eer = calculate_roc(temp_loss_tensor, step=0.001)
        EERs[:, batch_size - 1] = eer[0, :]
        if plot_roc:
            plot_roc(roc1, roc2, eer)

        print("EER MIN: {0}".format(eer[0]))

    return EERs


def calculate_min_windows_loss(loss_tensor, batch_size):
    return reduce_dimension(loss_tensor, batch_size, 'min')


def calculate_max_windows_loss(loss_tensor, batch_size):
    return reduce_dimension(loss_tensor, batch_size, 'max')


def calculate_mean_windows_loss(loss_tensor, batch_size):
    return reduce_dimension(loss_tensor, batch_size, 'mean')


def reduce_dimension(loss_tensor, batch_size, method):
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
        if method == 'min':
            temp_loss_tensor[:, :, x] = np.min(loss_batch, axis=2)
        if method == 'max':
            temp_loss_tensor[:, :, x] = np.max(loss_batch, axis=2)
        if method == 'mean':
            temp_loss_tensor[:, :, x] = np.mean(loss_batch, axis=2)
        x += 1

    return temp_loss_tensor


def calculate_roc(loss_tensor, step=0.001, last_index=1, first_index=0):
    last_index += step
    first_index -= step
    n_models = np.shape(loss_tensor)[0]

    thresholds = np.arange(first_index, last_index + step, step)
    roc1 = np.zeros((2, len(thresholds), n_models))
    roc2 = np.zeros((2, len(thresholds), n_models))
    for i, threshold in zip(range(len(thresholds)), thresholds):
        if i%100 == 0:
            print(".", end="")
        scores = compute_classification_scores(loss_tensor, threshold)
        for j in range(n_models):
            roc1[0, i, j] = scores[j]["FNR"]
            roc1[1, i, j] = scores[j]["FPR"]

            roc2[0, i, j] = scores[j]["TPR"]
            roc2[1, i, j] = scores[j]["FPR"]
            # print("ACC {0} - "+str(scores[0]["ACC"]), end =";")
            # print("TP {0} - " + str(scores[0]["TP"]), end=";")
            # print("FP {0} - " + str(scores[0]["FP"]), end=";")
            # print("FN {0} - " + str(scores[0]["FN"]), end=";")
            # print("TN {0} - " + str(scores[0]["TN"]))

    print("End of ROC")

    return roc1, roc2, scores


def calculate_equal_error_rate(roc):
    n_models = np.shape(roc)[2]
    eers = np.zeros(n_models) - 1

    for j in range(n_models):
        candidate = roc[0, :, j] - roc[1, :, j]
        candidate_index = np.argmin(candidate[candidate > 0])
        eers[j] = roc[0, candidate_index, j]

        if candidate[candidate_index] != 0:
            eers[j], new_fpr_x, new_fpr_y = find_eer_by_interpolation(roc, j)

        print("EER for model {0}: {1}".format(j+1, eers[j]))

    return eers


def find_eer_by_interpolation(roc, j, plot_interpolation=False):
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
    fnr_y = roc[0, array_indexes, j]
    fpr_x = roc[1, array_indexes, j]

    x_interpolation = interpolate.interp1d(array_indexes, fpr_x)
    y_interpolation = interpolate.interp1d(array_indexes, fnr_y)

    candidate = np.array([1])
    candidate_index = np.argmin(candidate)
    array_size = 50
    new_fnr_y = []
    patience = 0
    patience_max = 20

    while candidate[candidate_index] > 0.00001 and patience < patience_max:
        patience += 1
        new_array_indexes = np.linspace(array_indexes[0], array_indexes[-1], num=array_size)
        try:
            new_fpr_x = x_interpolation(new_array_indexes)
            new_fnr_y = y_interpolation(new_array_indexes)

            candidate = abs(new_fpr_x - new_fnr_y)
            candidate_index = np.argmin(candidate)

            array_size *= 2

        except:
            plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
            plt.scatter(fpr_x, fnr_y, marker='.')
            plt.show()
            break

    if plot_interpolation:
        plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
        plt.scatter(fpr_x, fnr_y, marker='.')
        plt.show()

    eer = new_fnr_y[candidate_index]
    if patience == patience_max:
        print("Patience is exausted when searching for eer, minimum found: "+str(candidate[candidate_index]))
    if math.isnan(eer):
        print("NaN detected in ERR calculations:")
        print(new_fpr_x)
        print(new_fnr_y)

    return eer, new_fpr_x, new_fnr_y


def plot_roc(roc1, roc2, eer, n_signals=20):
    name = 'gnuplot'
    cmap = plt.get_cmap(name)
    cmap_list = [cmap(i) for i in np.linspace(0, 1, n_signals)]
    fig_1 = "ROC False Negative Rate/False Positive Rate"
    fig_2 = "ROC True Positive Rate/False Positive Rate"
    f, (ax1, ax2) = plt.subplots(1, 2)

    for j in range(np.shape(roc1)[2]):
        ax1.subplot().figure(fig_1)
        ax1.scatter(roc1[1, :, j], roc1[0, :, j], marker='.', color=cmap_list[j], label='Signal #{0}'.format(j))
        ax1.plot(roc1[1, :, j], roc1[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal #{0}'.format(j))

        ax2.figure(fig_2)
        ax2.scatter(roc2[1, :, j], roc2[0, :, j], marker='.', color=cmap_list[j], label='Signal #{0}'.format(j))
        ax2.plot(roc2[1, :, j], roc2[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal #{0}'.format(j))

    N_Signals = np.shape(eer)[1]
    for signal in range(N_Signals):
        ax1.figure(fig_1)
        ax1.plot(eer[1, signal], eer[0, signal], color="#990000", marker='o', alpha=0.2)

    ax1.plot([0, 1], [0, 1], color="#990000", alpha=0.2)
    ax1.ylabel("False Negative Rate")
    ax1.xlabel("False Positive Rate")
    ax1.ylim([0, 1])
    ax1.xlim([0, 1])
    ax1.legend()

    ax2.ylabel("True Positive Rate")
    ax2.xlabel("False Positive Rate")
    ax2.ylim([0, 1])
    ax2.xlim([0, 1])
    ax2.legend()
    ax2.show()


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
        min_max[0] -= (len(roc[0, :, j]) + 1 + interval)
        min_max[1] = len(roc[0, :, j]) - 1

    return min_max


def compute_classification_scores(loss_tensor, threshold=0.1):
    n_Signals = np.shape(loss_tensor)[1]
    scores = list(range(n_Signals))

    prediction_matrix, prediction_tensor = calculate_prediction_matrix(loss_tensor, threshold)
    confusion_tensor = get_classification_confusion(prediction_tensor)

    # [TP,FN]
    # [FP,TN]
    for i in list(range(n_Signals)):
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


def get_classification_confusion(signal_predicted_tensor):
    signal_list = np.arange(np.shape(signal_predicted_tensor)[0])
    confusion_tensor = np.zeros((len(signal_list), 2, 2))

    for signal in signal_list:
        # [TP,FN]
        # [FP,TN]
        classified_matrix = signal_predicted_tensor[signal, :, :]
        confusion_tensor[signal, 0, 0] = len(np.where(classified_matrix[signal, :] == 1)[0]) # TP
        confusion_tensor[signal, 0, 1] = len(np.where(classified_matrix[signal, :] == 0)[0]) # FN

        confusion_tensor[signal, 1, 0] = len(np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 1)[0]) # FP
        confusion_tensor[signal, 1, 1] = len(np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 0)[0]) # TN

    return confusion_tensor


def calculate_prediction_matrix(loss_tensor, threshold = 1):
    threshold_layer = np.ones((np.shape(loss_tensor)[1],np.shape(loss_tensor)[2])) * threshold
    normalized_loss_tensor = np.zeros((np.shape(loss_tensor)[0]+1,np.shape(loss_tensor)[1],np.shape(loss_tensor)[2]))
    predicted_tensor = np.zeros_like(loss_tensor)

    for j in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:-1, j, :] = loss_tensor[:, j, :] - np.min(loss_tensor[:, j, :], axis=0)
        normalized_loss_tensor[:-1, j, :] = normalized_loss_tensor[:-1, j, :] / np.max(loss_tensor[:, j, :], axis=0)

    normalized_loss_tensor[-1, :, :] = threshold_layer

    for i in range(np.shape(loss_tensor)[0]):
        predicted_tensor[i, : , :] = np.argmin(normalized_loss_tensor[[-1, i], :, :], axis=0)

    predicted_matrix = np.argmin(normalized_loss_tensor, axis=0)

    return predicted_matrix, predicted_tensor


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


def process_EER(loss_tensor, max_n_windows, iterations=20, plot_models_eer=False, plot_iterations_eer=False):
    n_models = np.shape(loss_tensor)[0]
    EERs = np.zeros((n_models, max_n_windows, iterations))
    time = np.arange(1, max_n_windows + 1) * 0.33

    for iteration in range(iterations):
        roc = calculate_roc(loss_tensor, step=0.0001)
        EERs[:, :, iteration] = calculate_eer(loss_tensor, max_n_windows=120)

        if plot_models_eer:
            labels = ["ECG ".format(i) for i in range(1, n_models + 1)]
            plot_EERs(EERs[:, :, iteration], time, labels, "Mean EER for different models", "_X_model")

    if plot_iterations_eer:
        labels = ["ITER ".format(i) for i in range(1, iterations + 1)]
        plot_EERs(np.mean(EERs, axis=0).T, time, labels, "Mean EER for different iterations", "_X_iter")

    return EERs


def save_loss_tensor(filename, loss_tensor, signal_models):
    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signal_models)

    return loss_tensor