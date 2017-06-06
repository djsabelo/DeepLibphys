import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
from itertools import repeat
import time
import seaborn
import math
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import DeepLibphys.utils.functions.database as db


def calculate_eers(batch_size, loss_tensor, iteration):
    temp_loss_tensor = calculate_min_windows_loss(loss_tensor, batch_size)
    if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
        return

    roc1, roc2, scores, eer_min = calculate_roc(temp_loss_tensor, step=0.001)
    # plot_roc(roc1, roc2, eer_min)
    print("EER MIN of iteration {0}: ,{0}".format(iteration, eer_min[0]))

    return [eer_min[0, :], iteration, batch_size, scores, roc1, roc2]

def descompress_data(data):
    EERs, iteration, batch_size, scores, roc1, roc2 = [],[],[],[],[],[]
    for line in data:
        EERs.append(line[0])
        iteration.append(line[1])
        batch_size.append(line[2])
        scores.append(line[3])
        roc1.append(line[4])
        roc2.append(line[5])

    new_data = {
        "EERs": EERs,
        "iteration": iteration,
        "batch_size": batch_size,
        "scores": scores,
        "roc1": roc1,
        "roc2": roc2,
        "best": np.argmin(np.mean(EERs, axis=0))
    }
    return EERs, iteration, batch_size, scores, roc1, roc2, new_data

def process_alternate_eer(loss_tensor, iterations=10, savePdf=True, SNR=1, name=""):
    all_data = []
    batch_size_array = np.arange(1, 60)
    N_Signals = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = np.arange(1, 60) * 0.33
    all_data = []
    for iteration in range(iterations):
        pool = Pool()
        data = pool.starmap(calculate_eers, zip(batch_size_array, repeat(loss_tensor), repeat(iteration)))
        EERs, iteration, batch_size, scores, roc1, roc2, save_data = descompress_data(data)
        EERs = np.array(EERs)

        labels = ["ECG {0}".format(i) for i in range(1, N_Signals+1)]

        plot_errs(EERs, seconds, labels, "Mean EER for SNR of {0}".format(SNR),
                  "_SNR_{0}{1}_{2}".format(SNR, name, iteration), savePdf=savePdf)

        all_data.append(save_data)
    np.savez("../data/validation/ALL_DATA_SNR_{0}{1}.npz".format(SNR, name), all_data = all_data)
    labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    plot_errs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_SNR_ITER_{0}{1}".format(SNR, name), savePdf=savePdf)

    return np.mean(EERs, axis=0).T


def process_eer(loss_tensor, iterations=10, savePdf=True, SNR=1, name=""):
    all_data = []
    batch_size_array = np.arange(1, 60)
    N_Signals = np.shape(loss_tensor)[0]
    N_Total_Windows = np.shape(loss_tensor)[2]
    EERs = np.zeros((N_Signals, len(batch_size_array), iterations))
    seconds = np.arange(1, 60) * 0.33
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

        plot_errs(EERs[:, :, iteration], seconds, labels, "Mean EER for SNR of {0}".format(SNR),
                  "_SNR_{0}{1}_{2}".format(SNR, name, iteration), savePdf=savePdf)

    np.savez("../data/validation/ALL_DATA_SNR_{0}{1}.npz".format(SNR, name), all_data = all_data)
    labels = ["ITER {0}".format(i) for i in range(1, iterations + 1)]
    plot_errs(np.mean(EERs, axis=0).T, seconds, labels, "Mean EER for different iterations", "_SNR_ITER_{0}{1}".format(SNR, name), savePdf=savePdf)

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

        plt.annotate("EER MIN STD = {0:.4f}".format(np.std(EER)),
                     xy=(indexi, indexj - step))

        plt.annotate("TIME FOR MIN MEAN = {0:.4f}".format(time[index_min]),
                     xy=(indexi, indexj - step * 2))

        plt.annotate("TIME FOR MIN STD = {0:.4f}".format(mean_EERs[index_min]),
                     xy=(indexi, indexj - step * 3))
    else:
        for i, EER in zip(range(len(EERs)), EERs):
            index_min = np.argmin(EER)
            plt.annotate("EER MIN = {0:.3f}".format(EER[index_min]),
                         xy=(time[index_min], EER[index_min] + 0.01), color=cmap(i / len(EERs)), ha='center')

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

        scores[i] = {"TP": TP,
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

    remake = False
    candidate_index = 1
    print("end")
    for j in range(np.shape(loss_tensor)[0]):
        candidate = roc1[0, :, j] - roc1[1, :, j]
        candidate_index = np.argmin(candidate[candidate > 0])
        eer[0, j] = roc1[0, candidate_index, j]
        eer[1, j] = roc1[1, candidate_index, j]

        if candidate[candidate_index] != 0:
            eer_j, new_fpr_x, new_fpr_y = find_eeq(roc1, j)
            eer[0, j] = eer_j
            eer[1, j] = eer_j
            print(eer_j)

    return roc1, roc2, scores, eer


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


GRU_DATA_DIRECTORY = "../data/trained/"
N_Windows = 6000
W = 256
noise_filename = "../data/ecg_noisy_signals.npz"
npzfile = np.load(noise_filename)
processed_noise_array, SNRs = npzfile["processed_noise_array"], npzfile["SNRs"]


loss_quartenion = [np.load("../data/CLEAN_1024" + ".npz")["loss_tensor"]]
description = "NOISY_INDEX_WITH_NOISE_"

signals_models = [db.ecg_SNR_12, db.ecg_SNR_11, db.ecg_SNR_9, db.ecg_SNR_8]

SNRs = SNRs[[0, 1, 3, 4]].tolist()
[loss_quartenion.append(np.load('../data/validation/{0}{1}'.format(description, snr) + ".npz")["loss_tensor"]) for
 snr, signal_models in zip(SNRs, signals_models)]

SNRs = ["NA"] + SNRs
labels = [str(SNR) for SNR in SNRs]
time = np.arange(1, 60) * 0.33
mean_EERs = np.zeros((len(SNRs), len(time)))

mean_EERs = []
for SNR, loss_tensor in zip(SNRs, loss_quartenion):
    mean_EERs.append(process_alternate_eer(loss_tensor, iterations=10, savePdf=False, SNR=SNR, name=str(SNR)))
#
# np.savez("eers.npz", EERs=mean_EERs)
# def process_parallel_EERs(n, SNR, loss_tensor):
#     print("SNR of " + str(SNR))
#     return process_eer(loss_tensor, iterations=1, savePdf=False, SNR=SNR, name=description)
#
# pool = Pool()
# EERs = pool.starmap(process_parallel_EERs, zip(range(len(SNRs)), SNRs, loss_quartenion))
# np.savez("eers.npz", EERs=EERs)
seconds = np.arange(1, 60) * 0.33
EERs = np.load("eers.npz")["EERs"]
EERs = np.reshape(EERs, (np.shape(EERs)[0], np.shape(EERs)[2]))
plot_errs(EERs, seconds, labels, "Mean EER for ALL SNR", "all_SNR", savePdf=True, plot_mean=False)