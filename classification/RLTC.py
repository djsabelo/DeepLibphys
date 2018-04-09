# Relative Loss Threshold Classification

import numpy as np

from DeepLibphys.utils.functions.common import *
import DeepLibphys.models.LibphysMBGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
from multiprocessing import Pool
from itertools import repeat
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import time
import theano
import theano.tensor as T
import seaborn
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib



# theano experiments

def calculate_theano_prediction_tensor(loss_tensor, threshold=1, lvl_acceptance=5):

    predicted_tensor = np.zeros_like(loss_tensor)
    normalized_loss_tensor = normalize_tensor(loss_tensor, lvl_acceptance=lvl_acceptance)
    normalized_tensor = theano.shared(name='normalized_tensor',
                                      value=normalized_loss_tensor.astype(theano.config.floatX))





    if isinstance(threshold, list) or isinstance(threshold, np.ndarray):
        threshold = np.array(threshold)
        for i, thresh in enumerate(threshold):
            predicted_tensor[i, np.where(normalized_loss_tensor[i] < thresh)] = 1
    else:
        predicted_tensor[np.where(normalized_loss_tensor < threshold)] = 1

    return predicted_tensor

def calculate_theano_eer(batch_size):
    batch_size_array = np.arange(1, batch_size)
    T.where
    EERs, thresholds = calculate_err_in_different_batches(loss_tensor, batch_size_array, decimals)

    [eer, best_thresholds, indexes], updates = theano.scan(
        calculate_eers,
        sequences=batch_size_array,
        outputs_info=[dict(initial=T.zeros(batch_size, 40)),
                      dict(initial=T.zeros((batch_size, 40))),
                      dict(initial=T.zeros((batch_size, 40)))])
    return


def normalize_theano_tensor(loss_tensor, lvl_acceptance):
    normalized_loss_tensor = (loss_tensor - np.min(loss_tensor, axis=0)) / \
                             np.max(loss_tensor - np.min(loss_tensor, axis=0), axis=0)
    Mshape = (np.shape(normalized_loss_tensor)[1],np.shape(normalized_loss_tensor)[2])
    mean_values = np.mean(normalized_loss_tensor, axis=2)
    cropx = np.sort(mean_values)[:, lvl_acceptance]

    normalized_tensor = T.dtensor3('normalized_tensor')# theano.shared(name='normalized_tensor', value=normalized_loss_tensor.astype(theano.config.floatX))
    crops = T.dvector('crops') #theano.shared(name='crops', value=crops.astype(theano.config.floatX))

    def crop_normalized_tensor(last, normalized_matrix, crop):
        # x,y = T.dmatrices('x','y')
        # v = T.where(, x, y)
        normalized_matrix[T.lt(normalized_matrix, crop)] = crop
        return normalized_matrix/T.max(normalized_matrix)

    n_l_t, updates = theano.scan(
        crop_normalized_tensor,
        sequences=[normalized_tensor, crops],
        outputs_info=[None, dict(initial=T.zeros_like(normalized_tensor))])

    normalizexx_loss_tensor = theano.function([normalized_tensor, crops], n_l_t)

    normalized_loss_tensor = normalizexx_loss_tensor(normalized_loss_tensor, cropx)
    return np.nan_to_num(normalized_loss_tensor)

def loss_worker(x_matrix, y_matrix, signal):
    return

# LOSS FUNCTION
def calculate_loss_tensor(Total_Windows, W, signals_models, signals, mean_tol=10000, overlap=0.33,
                          batch_percentage=0, mini_batch=256, std_tol=10000, X_matrix=None, Y_matrix=None,
                          min_windows=100):

    if X_matrix is None and Y_matrix is None:
        prepare_data = True
        X_matrix = []
        Y_matrix = []
    else:
        prepare_data = False



    sizes = []
    removex = []
    for signal, model_info, i in zip(signals, signals_models, range(len(signals))):
        signal2model = Signal2Model(model_info.dataset_name, "", signal_dim=model_info.Sd,
                                    hidden_dim=model_info.Hd, batch_size=Total_Windows,
                                    window_size=W)
        if type(signal[0]) is np.int64 or type(signal[0]) is np.float64:
            signal = [signal]

        if prepare_data:
            X_list, Y_list = prepare_test_data(signal, signal2model, overlap=overlap,
                                               batch_percentage=batch_percentage, mean_tol=mean_tol, std_tol=std_tol,
                                               randomize=False)

            if np.shape(X_list)[0] >= min_windows:
                X_matrix.append(X_list)
                Y_matrix.append(Y_list)
                sizes.append(np.shape(X_list)[0])
            else:
                removex.append(i)
        else:
            print(np.shape(X_matrix[i]))
            sizes.append(np.shape(X_matrix[i])[0])

    removex.sort(reverse=True)
    [signals_models.pop(rem) for rem in removex]

    print(np.shape(X_matrix))

    max_windows = np.min(np.array(sizes))
    for t, test_signal in enumerate(X_matrix):
        X_matrix[t] = test_signal[:max_windows]
        Y_matrix[t] = Y_matrix[t][:max_windows]
    print(np.shape(X_matrix))

    X_matrix, Y_matrix = np.array(X_matrix), np.array(Y_matrix)
    max_windows = max_windows - (max_windows % mini_batch)
    print("Number of Windows: {0} of {1}".format(max_windows, np.max(np.array(sizes))))

    windows = np.arange(0, max_windows, mini_batch)
    print(windows)
    N_Models = len(signals_models)
    N_Signals = len(X_matrix)

    loss_tensor = np.zeros((N_Models, N_Signals, max_windows))

    print("Loading model...")
    model_info = signals_models[0]

    signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                hidden_dim=model_info.Hd, mini_batch_size=mini_batch)
    model = GRU.LibphysMBGRU(signal2Model)
    times = []
    for m, model_info in zip(range(len(signals_models)), signals_models):
        model.model_name = model_info.dataset_name
        model.load(dir_name=model_info.directory)
        print("Processing Model " + model_info.name + " - time: " + str(model.train_time))

        for s in range(N_Signals):
            print("Calculating loss for ECG " + str(s + 1), end=';\n ')
            for w in windows:
                x_test = X_matrix[s, w:w + mini_batch, :]
                y_test = Y_matrix[s, w:w + mini_batch, :]
                tic = time.time()
                loss_tensor[m, s, w:w + mini_batch] = np.asarray(model.calculate_mse_vector(x_test, y_test))
                times.append(time.time() - tic)

    times = np.array(times)
    print(np.size(loss_tensor, 2))
    print("Statistics: \n Mean time: {0}; \n Std time: {1}; Max time: {2}; Min Time: {3}".format(
        np.mean(times), np.std(times), np.max(times), np.min(times)))
    return loss_tensor


def get_or_save_loss_tensor(full_path, N_Windows=None, W=None, models=None, test_signals=None, force_new=False,
                            mean_tol=100000, overlap=0.33, batch_percentage=0, mini_batch=128, std_tol=1000,
                            X_matrix=None, Y_matrix=None, min_windows=200):
    if not os.path.isfile(full_path) or force_new:
        loss_tensor = calculate_loss_tensor(N_Windows, W, models, test_signals, mean_tol=mean_tol, overlap=overlap,
                                            batch_percentage=batch_percentage, mini_batch=mini_batch, std_tol=std_tol,
                                            X_matrix=X_matrix, Y_matrix=Y_matrix, min_windows=min_windows)
        np.savez(full_path, loss_tensor=loss_tensor)
        print("Saving to {0}".format(full_path))
    elif os.path.isfile(full_path + '.npz'):
        loss_tensor = np.load(full_path + '.npz')["loss_tensor"]
    else:
        loss_tensor = np.load(full_path)["loss_tensor"]

    return loss_tensor


def calculate_batch_min_loss(loss_z_tensor, batch_size):
    N_Models = np.shape(loss_z_tensor)[0]
    N_Signals = np.shape(loss_z_tensor)[1]
    N_Total_Windows = np.shape(loss_z_tensor)[2] - (np.shape(loss_z_tensor)[2] % batch_size)
    batch_indexes = np.arange(0, N_Total_Windows, batch_size)
    N_Batches = len(batch_indexes)
    if N_Batches == 0:
        return -1

    temp_loss_tensor = np.zeros((N_Models, N_Signals, N_Batches))
    print(batch_size, end=" - ")

    for x, i in enumerate(batch_indexes):
        loss_batch = loss_z_tensor[:, :, i:i + batch_size]
        temp_loss_tensor[:, :, x] = np.min(loss_batch, axis=2)

    return temp_loss_tensor


def calculate_batch_mean_loss(loss_tensor, batch_size):
    N_Models = np.shape(loss_tensor)[0]
    N_Signals = np.shape(loss_tensor)[1]
    N_Total_Windows = np.shape(loss_tensor)[2] - (np.shape(loss_tensor)[2] % batch_size)
    batch_indexes = np.arange(0, N_Total_Windows, batch_size)
    N_Batches = len(batch_indexes)
    if N_Batches == 0:
        return -1

    temp_loss_tensor = np.zeros((N_Models, N_Signals, N_Batches))
    print(batch_size, end=" - ")

    for x, i in enumerate(batch_indexes):
        loss_batch = loss_tensor[:, :, i:i + batch_size]
        temp_loss_tensor[:, :, x] = np.mean(loss_batch, axis=2)

    return temp_loss_tensor


def filter_loss_tensor(signals, loss_tensor, all_models_info, W, min_windows=512, overlap=0.33, max_tol=0.8,
                       std_tol=0.5, batch_percentage=0, already_cut=False):
    model_indexes = []
    list_of_windows_indexes = []
    number_of_windows = []
    do_it = True
    try:
        signals[0][0][0]
    except:
        do_it = False

    for i, model_info in enumerate(all_models_info):
        signal2model = Signal2Model(model_info.dataset_name, "", signal_dim=model_info.Sd,
                                    hidden_dim=model_info.Hd, batch_size=np.shape(loss_tensor)[2],
                                    window_size=W)
        if do_it:
            s = signals[i]
        else:
            s = [signals[i]]
        indexes, _, _, _, _ = get_clean_indexes(s, signal2model, overlap=overlap, max_tol=max_tol,
                                                std_tol=std_tol, already_cut=already_cut)
        indexes = np.array(indexes)
        if batch_percentage > 0:
            all_indexes, _, _, _, _ = get_clean_indexes(signals[i], signal2model, overlap=overlap, max_tol=0.7)
            all_indexes = all_indexes[int(len(all_indexes)*batch_percentage):]

            new_indexes = []
            for ind in indexes[indexes >= all_indexes[0]]:
                if np.any(all_indexes == ind):
                    new_indexes.append(ind)

            indexes = new_indexes

        if len(indexes) >= min_windows + 20:
            model_indexes.append(i)
            list_of_windows_indexes.append(indexes)
            number_of_windows.append(len(indexes))

    li = 0
    new_models_indexes = []
    indexes_2_remove = []
    Total = min(number_of_windows)
    loss_list = np.zeros((len(model_indexes), len(model_indexes), Total - 1))
    for i, inds in zip(model_indexes, list_of_windows_indexes):
        first_index = 0
        if len(inds) > Total + 20:
            first_index = np.random.random_integers(0, len(inds) - Total - 20, 1)[0]
        end_index = first_index + Total - 1

        if inds[end_index] > np.shape(loss_tensor)[2]:
            first_index = 0
            end_index = first_index + Total - 1

        if inds[end_index] < np.shape(loss_tensor)[2]:
            L = loss_tensor[i][model_indexes][:, inds[first_index:end_index]]
            if np.shape(L)[1] > 0:
                loss_list[li] = L
                new_models_indexes.append(i)
        if np.alltrue(loss_list[li] == np.zeros_like(loss_list[li])):
            indexes_2_remove.append(li)

        li += 1

    if indexes_2_remove != []:
        mask = mask_without_indexes(loss_list, indexes_2_remove)
        loss_list = loss_list[mask][:, mask]

    model_indexes = new_models_indexes
    print("Rejected {0} people, total of windows = {1}".format(len(all_models_info) - len(model_indexes), Total))

    if model_indexes != []:
        return loss_list, np.array(all_models_info)[np.array(model_indexes)]
    else:
        print("All models were rejected")
        return [], []


# IDENTIFICATION


def identify_biosignals(loss_x_tensor, models_info, batch_size=1, thresholds=1, name="", norm=False):
    if batch_size > 1:
        temp_loss_tensor = calculate_batch_min_loss(loss_x_tensor, batch_size)
    else:
        temp_loss_tensor = loss_x_tensor

    model_labels = np.asarray(np.zeros(len(models_info) * 2, dtype=np.str), dtype=np.object)
    model_labels[list(range(1, len(models_info) * 2, 2))] = [model_info.name for model_info in models_info]
    signal_labels = model_labels

    sinal_predicted_matrix = calculate_prediction_matrix(temp_loss_tensor, thresholds)

    plot_confusion_matrix(sinal_predicted_matrix, signal_labels, model_labels, no_numbers=False, norm=True, name=name)


def normalize_tensor(loss_tensor, lvl_acceptance=None):
    timex = []
    timex.append(time.time())
    normalized_loss_tensor = (loss_tensor - np.min(loss_tensor, axis=0)) / \
                             np.max(loss_tensor - np.min(loss_tensor, axis=0), axis=0)

    timex.append(time.time())
    if lvl_acceptance is not None:
        mean_values = np.mean(normalized_loss_tensor, axis=2)
        for m, model_values in enumerate(mean_values):
            crop = np.sort(model_values)[lvl_acceptance]
            normalized_loss_tensor[m][np.where(normalized_loss_tensor[m] >= crop)] = crop

        normalized_loss_tensor = normalized_loss_tensor / np.max(normalized_loss_tensor, axis=0)

    timex.append(time.time())
    normalized_loss_tensor = np.nan_to_num(normalized_loss_tensor)
    timex.append(time.time())
    # print("Total Time: {0} s; ".format(timex[-1] - timex[0]), end="")
    #
    # [print("Time {0}: {1} s; ".format(i, t - timex[i]), end=";") for i, t in enumerate(timex[1:])]
    # print("")
    # mean_values = np.mean(normalized_loss_tensor, axis=2)
    # for model_values in mean_values:
    #     sorted = np.sort(model_values)
    #     plt.plot(sorted[:15])
    # plt.show()
    return normalized_loss_tensor


def calculate_prediction_tensor(loss_tensor, threshold=1, lvl_acceptance=5):
    predicted_tensor = np.zeros_like(loss_tensor)
    normalized_loss_tensor = normalize_tensor(loss_tensor, lvl_acceptance=lvl_acceptance)
    if isinstance(threshold, list) or isinstance(threshold, np.ndarray):
        threshold = np.array(threshold)
        for i, thresh in enumerate(threshold):
            predicted_tensor[i, np.where(normalized_loss_tensor[i] < thresh)] = 1
    else:
        predicted_tensor[np.where(normalized_loss_tensor < threshold)] = 1

    return predicted_tensor



def calculate_prediction_matrix(loss_tensor, threshold=1, lvl_acceptance=None):
    normalized_loss_tensor = normalize_tensor(loss_tensor, lvl_acceptance)
    predicted_matrix = np.zeros_like(loss_tensor[:, :, 0])
    if isinstance(threshold, list) or isinstance(threshold, np.ndarray):
        threshold = np.array(threshold)
        for i, thresh in enumerate(threshold):
            predicted_matrix[i, np.where(normalized_loss_tensor[i] < thresh)] += 1

            predicted_matrix = np.sum(normalized_loss_tensor, axis=0)
    else:
        for i, predicted_line in enumerate(np.argmin(normalized_loss_tensor, axis=0)):
            for j, predicted_sample in enumerate(predicted_line):
                predicted_matrix[predicted_sample, i] += 1

    return predicted_matrix


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
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j)

    return sinal_predicted_matrix


# EQUAL ERROR RATE


def process_eers(loss_tensor, W, full_path, name, save_pdf=True, batch_size=120, fs=250, decimals=4,  force_new=True):
    batch_size_array = np.arange(1, batch_size)
    seconds = batch_size_array * 0.33 * W / fs
    directory = full_path
    full_path = full_path + "/" + name + "_EER.npz"
    if not os.path.isfile(full_path) or force_new:
        EERs, thresholds, candidate_indexes = calculate_err_in_different_batches(loss_tensor, batch_size_array, decimals)
        np.savez(full_path, EERs=EERs, thresholds=thresholds, candidate_indexes=candidate_indexes)
    else:
        file = np.load(full_path)
        EERs, thresholds, candidate_indexes = file["EERs"], file["thresholds"], file["candidate_indexes"]

    if save_pdf:
        labels = ["ECG {0}".format(i) for i in range(1, np.shape(loss_tensor)[0] + 1)]
        plot_errs(EERs, seconds, labels, directory, name, "Mean EER", savePdf=save_pdf, plot_mean=True)

    return EERs, thresholds, batch_size_array


def calculate_err_in_different_batches(loss_tensor, batch_size_array, decimals=4):
    pool = Pool(12)
    x = np.round(time.time() * 10 ** 7) / np.random.randint(10 ** 7, (10 ** 8 - 1), 1)
    if x < 2 ** 32 - 1:
        np.random.seed(int(x))

    # for batch in batch_size_array:
    #     data = calculate_eers(batch, loss_tensor, decimals)
    data = pool.starmap(calculate_eers, zip(batch_size_array, repeat(loss_tensor),
                                                                           repeat(decimals)))

    eers = np.array([line[0] for line in data]).T
    thresholds = np.array([line[1] for line in data]).T
    candidate_indexes = np.array([line[2] for line in data]).T
    pool.close()

    return eers, thresholds, candidate_indexes


def calculate_eers(batch_size, loss_tensor, decimals=4):
    print("batch_size = {0}".format(batch_size))

    temp_loss_tensor = calculate_batch_min_loss(loss_tensor, batch_size)
    if len(np.shape(temp_loss_tensor)) == 1 and temp_loss_tensor == -1:
        return

    eer, thresholds, candidate_index = calculate_smart_roc(temp_loss_tensor, decimals=decimals)

    print("EER MIN of batch_size = {0}, minimum EER: {1}".format(batch_size, np.min(eer)))

    return eer, thresholds, candidate_index


def calculate_smart_roc(loss, last_index=1, first_index=0, decimals=4, lvl_acceptance=15):
    last_index += 1 * 10 ** (-decimals)
    first_index -= 1 * 10 ** (-decimals)
    lvl_acceptance = lvl_acceptance if lvl_acceptance < len(loss) else len(loss)-1
    thresholds = np.unique(np.round(normalize_tensor(loss, lvl_acceptance), decimals))
    thresholds = np.insert(thresholds, [0, len(thresholds)], [first_index, last_index])
    thresholds = np.sort(thresholds)
    n_thresholds = len(thresholds)
    while n_thresholds > 5000:
        thresholds = np.unique(np.round(normalize_tensor(loss, lvl_acceptance), decimals-1))
        thresholds = np.insert(thresholds, [0, len(thresholds)], [first_index, last_index])
        n_thresholds = len(thresholds)
        thresholds = np.sort(thresholds)

    print("Number of Thresholds: {0}".format(n_thresholds))
    N_Models = np.shape(loss)[0]
    N_Signals = np.shape(loss)[1]
    eer = np.zeros(N_Signals) - 1
    roc = np.vstack((np.zeros((1, n_thresholds, np.shape(loss)[0])),
                     np.ones((1, n_thresholds, np.shape(loss)[0]))))
    thresholds_out = np.zeros(N_Models)
    for i, thresh in enumerate(thresholds):
        if i % 100 == 0:
            print(".", end="")
        score = calculate_variables(loss, thresh, lvl_acceptance=lvl_acceptance)
        for j in range(N_Models):
            roc[0, i, j] = score[j][0]
            roc[1, i, j] = score[j][1]
        # print(thresh)
    # for j in range(10, N_Models):
    #     plt.figure(j)
    #     plt.plot(thresholds, roc[0, :, j])
    #     plt.plot(thresholds, roc[1, :, j])
    #     plt.show()
        # plot_roc(np.reshape(roc[:, :, j], (np.shape(roc)[0], np.shape(roc)[1], 1)))

    candidate_index = 0
    for j in range(N_Models):
        if np.logical_and(roc[0, 0, j] - roc[0, 1, j] == 1, roc[1, 0, j] - roc[1, 1, j] == -1):
            candidate_index = 1
            eer[j] = 0
            print("yei")
        elif np.logical_and(roc[0, 0, j] - roc[0, 1, j] == -1, roc[1, 0, j] - roc[1, 1, j] == 1):
            candidate_index = -1
            eer[j] = 1
            print("no yei")
        else:
            candidate = roc[0, :, j] - roc[1, :, j]
            candidate_index = np.argmin(candidate[candidate >= 0])
            eer[j] = roc[0, candidate_index, j]
            thresholds_out[j] = thresholds[candidate_index]

            if candidate[candidate_index] != 0:
                eer[j] = find_eeq(roc, j)

    print("end")
    # plot_roc(roc)
    return eer, thresholds_out, candidate_index


def calculate_variables(loss_tensor, threshold=0.1, lvl_acceptance=5):
    timex = []
    timex.append(time.time())
    prediction_tensor = calculate_prediction_tensor(loss_tensor, threshold, lvl_acceptance)
    timex.append(time.time())
    N_Signals = np.shape(prediction_tensor)[0]

    confusion_tensor = get_classification_confusion(prediction_tensor)
    timex.append(time.time())
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

        # FALL-OUT - FALSE POSITIVE RATE
        FPR = FP / (FP + TN)

        # MISS RATE - FALSE NEGATIVE RATE
        FNR = FN / (FN + TP)

        # ACCURACY
        ACC = (TP + TN) / (TP + TN + FP + FN)

        scores[i] = [FNR, FPR, ACC]

    timex.append(time.time())

    # print("Total Time: {0} s; ".format(timex[-1] - timex[0]), end="")

    # [print("Time {0}: {1} s; ".format(i, t - timex[i] ), end=";") for i, t in enumerate(timex[1:])]
    # print("")
    return scores


def get_classification_confusion(signal_predicted_tensor):
    signal_list = np.arange(np.shape(signal_predicted_tensor)[0])
    confusion_tensor = np.zeros((len(signal_list), 2, 2))

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


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[0][1], line2[0][0] - line2[0][1])
    ydiff = (line1[1][0] - line1[1][1], line2[1][0] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        print(line1[0], end="; ")
        print(line1[1], end="; ")
        print(line2[1])
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def find_eeq(roc, j):
    # interval = 10
    min_max = get_min_max(j, 1, roc)
    roc_min_max = roc[:, min_max, j]

    x, eer = line_intersection([min_max, roc_min_max[0, :]], [min_max, roc_min_max[1, :]])
    # print(roc[:,[0,-1],j], end = "; ")
    #
    # print(roc_min_max, end = "; ")
    # print(min_max, end = "; ")
    # print(eer)



    # roc_min_max = roc[:, min_max, j]
    #
    # print()
    #
    #
    # while (roc_min_max[0, 0] - roc_min_max[0, 1] == 0) or (roc_min_max[1, 0] - roc_min_max[1, 1] == 0):
    #     interval += 2
    #     if interval >= np.shape(roc)[1]:
    #         return 0, roc[0, -1, j], roc[1, -1, j]
    #     min_max = get_min_max(j, interval, roc)
    #     roc_min_max = roc[:, min_max, j]
    #
    # array_indexes = list(range(min_max[0], min_max[1]))
    # fnr_y = roc[0, array_indexes, j]
    # fpr_x = roc[1, array_indexes, j]
    #
    #
    # candidate = np.array([1])
    # candidate_index = np.argmin(candidate)
    # array_size = 30
    # new_fnr_y = []
    # patience = 0
    # PATIENCE_MAX = 40
    # start, end = array_indexes[0], array_indexes[-1]
    # while (candidate[candidate_index] > 0.00001 and patience < PATIENCE_MAX):
    #     patience += 1
    #     new_array_indexes = np.linspace(start, end, num=array_size)
    #     try:
    #         x_interpolation = interpolate.interp1d(array_indexes, fpr_x)
    #         y_interpolation = interpolate.interp1d(array_indexes, fnr_y)
    #         new_fpr_x = x_interpolation(new_array_indexes)
    #         new_fnr_y = y_interpolation(new_array_indexes)
    #
    #         # print("y: {0}, x: {1}".format(new_fnr_y, new_fpr_x), end="; ")
    #         candidate = abs(new_fpr_x - new_fnr_y)
    #         candidate_index = np.argmin(candidate)
    #         if candidate_index < 0:
    #             start = new_array_indexes[0]
    #             end = new_array_indexes[2]
    #         elif candidate_index > len(candidate):
    #             start = new_array_indexes[-3]
    #             end = new_array_indexes[-1]
    #         else:
    #             start = new_array_indexes[candidate_index - 1]
    #             end = new_array_indexes[candidate_index + 1]
    #
    #         fpr_x = new_fpr_x
    #         fnr_y = new_fnr_y
    #         array_indexes = new_array_indexes
    #         # print(new_fnr_y[candidate_index], end="..")
    #         # if len(new_array_indexes) != len(array_indexes):
    #
    #         # array_size *= 2
    #     except:
    #         plt.plot(new_fpr_x, new_fnr_y, new_fpr_x[candidate_index], new_fnr_y[candidate_index], 'ro')
    #         plt.scatter(fpr_x, fnr_y, marker='.')
    #         plt.show()
    #         break
    #
    # eer = new_fnr_y[candidate_index]
    # if patience == PATIENCE_MAX:
    #     print("Patience is exausted when searching for eer, minimum found: {0}"
    #           .format(candidate[candidate_index]))
    # if math.isnan(eer):
    #     print(new_fpr_x)
    #     print(new_fnr_y)

    return eer


def get_min_max(j, interval, roc):
    """

    :param j:
    :param interval:
    :param roc:
    :return:
    """
    diff_roc = roc[0, :, j] - roc[1, :, j]
    indexes = np.where(diff_roc < 0)[0]

    if len(indexes) == 0:
        index = 0
    # elif len(indexes) == 1 and indexes[0] == (len(diff_roc) - 1):
    #     index = indexes[0] - 1
    else:
        index = indexes[0]

    min_max = [index - interval, index]

    if roc[0, min_max[0], j] == 1 and roc[0, min_max[1], j] == 1:
        plt.plot(roc[0, :, j])
        plt.plot(roc[1, :, j])
        plt.show()

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


# PLOT

def plot_roc(roc1, roc2=None, eer=None):
    N_Signals = np.shape(roc1)[2]
    name = 'rainbow'
    cmap = plt.get_cmap(name)
    cmap_list = [cmap(i) for i in np.linspace(0, 1, N_Signals)]
    fig_1 = "ROC False Negative Rate/False Positive Rate"
    fig_2 = "ROC True Positive Rate/False Positive Rate"
    for j in range(N_Signals):
        plt.figure(fig_1)
        plt.scatter(roc1[1, :, j], roc1[0, :, j], marker='.', color=cmap_list[j])
        plt.plot(roc1[1, :, j], roc1[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal {0}'.format(j))
        if roc2 is not None:
            plt.figure(fig_2)
            plt.scatter(roc2[1, :, j], roc2[0, :, j], marker='.', color=cmap_list[j])
            plt.plot(roc2[1, :, j], roc2[0, :, j], color=cmap_list[j], alpha=0.3, label='Signal {0}'.format(j))

    if eer is not None:
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

    if roc2 is not None:
        plt.figure(fig_2)
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.legend()

    plt.show()
