
import csv
import numpy as np
from FindNoiseinECG.FeaturesANDClustering.WindowFeature import *
from FindNoiseinECG.FeaturesANDClustering.MultiDClustering import *
from FindNoiseinECG.GenerateThings.PlotSaver import *
from FindNoiseinECG.GenerateThings.PDFATextGen import *
from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import *


def get_info(filename="Data/cybhi_data_config.csv"):
    info_train, info_test = [], []
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(reader):
            if i > 0:
                row[1] = list(map(int, str.split(row[1], ",")))
                row[2] = bool(int(row[2]))

                if i % 2 != 0:
                    info_train.append(row)
                else:
                    info_test.append(row)

    return info_train, info_test


def get_windows_from_clusters(desirable_clusters, cluster_array, signal):
    relevant_clusters = np.zeros_like(cluster_array)
    for cluster in desirable_clusters:
        relevant_clusters[np.where(np.array(cluster_array) == cluster)[0]] = 1

    start_indexes = np.where(np.diff(relevant_clusters) == 1)[0]
    end_indexes = np.where(np.diff(relevant_clusters) == -1)[0]

    if len(start_indexes) == 0:
        start_indexes = np.array([0])
    elif end_indexes[0] < start_indexes[0]:
        start_indexes = np.insert(start_indexes, 0, 0)

    if len(end_indexes) == 0:
        end_indexes = np.array([len(relevant_clusters)-1])
    elif start_indexes[-1] > end_indexes[-1]:
        end_indexes = np.append(end_indexes, len(relevant_clusters)-1)

    return [TimeWindow(start, end, signal) for start, end in zip(start_indexes, end_indexes)]


def process_windows(windows, i):
    aux_signal = []
    index = 0
    indexes = []
    processed_signal = []
    for w, tw in enumerate(windows):
        print("\t Processing train window {0}/{1}".format(w+1, len(windows)))
        if len(windows) == 1:
            aux_signal = tw.tolist() if len(aux_signal) == 0 else tw.tolist() + aux_signal
        else:
            aux_signal = tw.tolist() if len(aux_signal) == 0 else tw.tolist() + aux_signal
        indexes.append([index, len(aux_signal)])
        index = len(aux_signal)

    aux_signal = process_dnn_signal(np.array(aux_signal), interval_size=256, window_rmavg=40, window_smooth=20,
                                window_rmstd=1000, confidence=0.01)
    for limits in indexes:
        processed_signal.append(aux_signal[limits[0]:limits[1]])
    plt.figure(i)
    plt.plot(aux_signal)
    if i == "test":
        plt.show()

    return processed_signal


# Process Clustering
def get_clusters(signal, fs, features_names, win, windows_size, clusters, plot=False):
    # Process Features
    print("Extracting features...")
    feature_matrix = np.array([
        WindowStat(signal, fs=fs, statTool=f, window_len=(win * fs) / w) for f, w in
        zip(features_names, windows_size)]
    ).transpose()
    X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(feature_matrix, n_clusters=clusters, Linkage='ward',
                                                                        Affinity='euclidean')
    stds = []

    for i in range(0, clusters):
        s = len(y_pred[np.where(y_pred == i)[0]].tolist())
        print("STD:" + str(np.std(signal[np.where(y_pred == i)[0]])))
        stds.append(np.std(signal[np.where(y_pred == i)[0]]))

    y_pred_aux = np.zeros_like(y_pred)
    f = 0
    lock = -1
    for i in sort_index_by_key(stds):
        # get_number_of_windows(clusters, y_pred)
        if (f == 0) and (len(np.where(y_pred == i)[0]) < len(y_pred) / 5):
            y_pred_aux[np.where(y_pred == i)[0]] = f + 1
            lock = 1
        else:
            y_pred_aux[np.where(y_pred == i)[0]] = f
            if lock == 1:
                f += 1
                lock = -1
            f += 1
    return y_pred_aux


features_names = ["std", "sum", "sum", "AmpDiff"]
windows_size = [256, 256, 64, 64]
i = 0
win = 512
fs = 250
clusters = 4

# Load Signal and Info
fileDir = "Data/CYBHi"
cyb_dir = RAW_SIGNAL_DIRECTORY + 'CYBHi/data/long-term'

# train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
#     get_cyb_dataset_raw_files(dataset_dir=CYBHi_ECG)
#
# np.savez(fileDir + "/raw_signals.npz", train_dates=train_dates, train_names=train_names,
#              train_signals=train_signals, test_dates=test_dates,
#              test_names=test_names, test_signals=test_signals)

infos_train, infos_test = get_info()
file = np.load(fileDir + "/raw_signals.npz")
train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
    file["train_dates"], file["train_names"], file["train_signals"], file["test_dates"],file["test_names"], \
    file["test_signals"]


# signals = []
# for info_train, info_test in zip(infos_train, infos_test):
#     [name, info_clusters_train, reject] = info_train[0:-1]
#     info_clusters_test = info_test[1]
#     train_signal = train_signals[np.where(train_names == name)[0]]
#     test_signal = train_signals[np.where(test_names == name)[0]]
#
#     if not reject and train_signal.size > 0:
#         try:
#             print(name + ": Cluster Train")
#             clusters_train = get_clusters(remove_noise(train_signal[0], moving_avg_window=40, smooth_window=20),
#                                           fs, features_names, win, windows_size, clusters)
#             print(name + ": Cluster Test")
#             clusters_test = get_clusters(remove_noise(test_signal[0], moving_avg_window=40, smooth_window=20),
#                                           fs, features_names, win, windows_size, clusters)
#
#             signals.append(Signal(train_signal, test_signal,
#                                   get_windows_from_clusters(info_clusters_train, clusters_train, train_signal),
#                                   get_windows_from_clusters(info_clusters_test, clusters_test, test_signal), name))
#         except:
#             print("ERROR")
#     else:
#         print("REJECTED - " + name)
#
#
# print("Saving!! 1st")
# np.savez(fileDir + "/signals.npz", signals=signals)
signals = np.load(fileDir + "/signals.npz")["signals"]

for i, signal_data in enumerate(signals):
    print("Signal " + signal_data.name)
    aux_signal = []
    indexes = []
    index = 0
    w = 0
    if type(signal_data.train_windows[0]) is TimeWindow:
        windows = [signal_data.train_signal[0][w.start_index:w.end_index] for w in signal_data.train_windows]
    else:
        windows = signal_data.train_signal

    signal_data.processed_train_windows = process_windows(windows, "train")

    if type(signal_data.test_windows[0]) is TimeWindow:
        windows = [signal_data.test_signal[0][w.start_index:w.end_index] for w in signal_data.test_windows]
    else:
        windows = signal_data.test_windows

    signal_data.processed_test_windows = process_windows(windows, "test")


print("Saving!! 2nd")
np.savez(fileDir + "/signals.npz", signals=signals)






# signals = np.load(fileDir + "/signals.npz")["signals"]
#
# for signal, name in zip(test_signals, test_names):
#     signal = remove_noise(signal, moving_avg_window=40, smooth_window=20)
#     time = np.arange(0, len(signal) / fs, 1 / fs)
#     fileName = "CYBHi_test_" + name
#     pp, ReportTxt = pdf_report_creator(fileDir, "Report_Color_" + fileName)
#
#     # Process Features
#     print("Extracting features...")
#     FeatureMatrixG = np.array([
#         WindowStat(signal, fs=fs, statTool=f, window_len=(win * fs) / w) for f, w in
#         zip(features_names, windows_size)]
#     ).transpose()
#     # plot linear features data
#     plotLinearData(time, FeatureMatrixG, signal, features_names, pp)
#     # Run Method
#     X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrixG, n_clusters=clusters, Linkage='ward',
#                                                             Affinity='euclidean')
#
#     print("Creating Predicted Array...")
#     Indexiser = []
#     stds = []
#     means = []
#
#     for i in range(0, clusters):
#         s = len(y_pred[np.where(y_pred == i)[0]].tolist())
#         stds.append(np.std(signal[np.where(y_pred == i)[0]]))
#         means.append(np.mean(abs(signal[np.where(y_pred == i)[0]])))
#         print("STD: {0}; MEAN: {1};".format(stds[-1], means[-1]))
#
#     y_pred_aux = np.zeros_like(y_pred)
#     f = 0
#     for i in sort_index_by_key(stds):
#         y_pred_aux[np.where(y_pred == i)[0]] = f
#         f += 1
#
#         # y_pred = sort_by_key(stds, y_pred)
#         # s = np.std(signal[np.where(y_pred == i)[0]])
#         # Indexiser.append(s)
#
#     # SigIndex = Indexiser.index(max(Indexiser))
#
#     # Prediction = np.ones(np.size(y_pred.tolist()))
#     # Prediction[np.where(y_pred == SigIndex)[0]] = 0
#
#     # plot clusters
#     # plotClusters(y_pred, signal, time, XPCA, clusters, pp)
#     plotClusters(y_pred_aux, signal, time, XPCA, clusters, pp)
#     pdf_text_closer(ReportTxt, pp)
