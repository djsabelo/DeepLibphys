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
from matplotlib.font_manager import FontProperties

GRU_DATA_DIRECTORY = "../data/trained/"
SYNTH_DIRECTORY = "../data/validation/SYNTH"
# SNR_DIRECTORY = "../data/validation/May_DNN_SNR_FANTASIA_1"


def calculate_loss_tensor(filename, Total_Windows, W, signals_models, signals=None, overlap=0.33):

    if Total_Windows > 256*4:
        n_windows = 256
    elif Total_Windows > 256:
        n_windows = int(Total_Windows/4)
    else:
        n_windows = Total_Windows

    windows = np.arange(0, Total_Windows - n_windows+1, n_windows)
    N_Windows = len(windows)
    N_Models = len(signals_models)
    Total_Windows = int(N_Windows * n_windows)
    N_Signals = len(signals)

    loss_tensor = np.zeros((N_Models, N_Signals, Total_Windows))

    X_matrix = np.zeros((N_Signals, Total_Windows, W))
    Y_matrix = np.zeros((N_Signals, Total_Windows, W))

    print("mini_batch: {0}, Total: {1}".format(n_windows, Total_Windows))
    i = 0

    for signal in signals:
        signal_test = segment_signal(signal[int(len(signal) * 0.33):], W, overlap)
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
            print("Calculating loss for Signal " + str(s + 1), end=';\n ')
            for w in windows:
                # index = w * n_windows
                x_test = X_matrix[s, w:w + n_windows, :]
                y_test = Y_matrix[s, w:w + n_windows, :]
                loss_tensor[m, s, w:w + n_windows] = np.asarray(model.calculate_mse_vector(x_test, y_test))
                # loss_tensor[m, s, w:w + n_windows] = np.asarray(model.calculate_loss_vector(x_test, y_test))

        # if not os.path.isdir(os.path.dirname(filename + ".npz")):
        #     os.mkdir(os.path.dirname(filename + ".npz"))

        np.savez(filename + ".npz",
                 loss_tensor=loss_tensor,
                 signals_models=signals_models)

    return loss_tensor


def plot_emg_confusion_matrix(confusion_matrix, labels_pred, labels_true, title='Confusion matrix' , cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=False, norm=False, N_Windows=None):
    # plt.tight_layout()

    N = np.shape(confusion_matrix)[0]
    fig, ax = plt.subplots()
    ax = prepare_confusion_error_plot(ax, confusion_matrix, labels_pred, labels_true, cmap, cmap_text, no_numbers,
                                       norm, N_Windows)

    # ax = prepare_confusion_pie(ax, confusion_matrix)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig("img/PREDICTED.eps", format='eps', dpi=50)
    plt.show()

def plot_confusion_matrix(confusion_matrix, labels_pred, labels_true, title='Confusion matrix' , cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=False, norm=False, N_Windows=None):
    # plt.tight_layout()

    N = np.shape(confusion_matrix)[0]
    fig, ax = plt.subplots()
    ax = prepare_confusion_matrix_plot(ax, confusion_matrix, labels_pred, labels_true, cmap, cmap_text, no_numbers,
                                       norm, N_Windows)

    # ax = prepare_confusion_pie(ax, confusion_matrix)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig("img/PREDICTED.eps", format='eps', dpi=50)
    plt.show()

def prepare_confusion_error_plot(ax, confusion_matrix, labels_pred, labels_true, cmap,
                                  cmap_text, no_numbers, norm, N_Windows):

    labels_predz = []
    for i in range(len(labels_pred)):
        labels_predz.append("")
        labels_predz.append(labels_pred[i])

        labels_truez = []
    for i in range(len(labels_true)):
        labels_truez.append("")
        labels_truez.append(labels_true[i])

    labels_pred, labels_true = labels_predz, labels_truez

    if norm:
        for i in range(np.shape(confusion_matrix)[0]):
            confusion_matrix[i] = confusion_matrix[i] - np.min(confusion_matrix[i])
            confusion_matrix[i] = confusion_matrix[i] / np.max(confusion_matrix[i])

    if norm:
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=N_Windows)
    elif N_Windows is not None:
        # norm = plt.Normalize(confusion_matrix.min(), confusion_matrix.max())
        # rgba = cmap(norm(confusion_matrix))
        # plt.imshow(rgba, interpolation='nearest', cmap=cmap, vmin=0, vmax=N_Windows)
        # for column, i in zip(confusion_matrix.T, range(np.shape(confusion_matrix)[1])):
        #     rgba[np.argmin(column), i, :] = 0, 1, 0, 0.2

        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=N_Windows)

    else:
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    # plt.tight_layout()
    kwargs = dict(size=30, fontweight='bold')
    plt.ylabel('Model', **kwargs)
    plt.xlabel('Signal', **kwargs)
    ax.set_xticks(np.arange(-0.5, np.shape(confusion_matrix)[1], 0.5))
    ax.set_yticks(np.arange(-0.5, np.shape(confusion_matrix)[0], 0.5))

    for i in range(len(ax.get_xgridlines())):
        if i % 2 == 0:
            ax.get_xgridlines()[i].set_linewidth(3)
        else:
            ax.get_xgridlines()[i].set_linewidth(0)

    for i in range(len(ax.get_ygridlines())):
        if i % 2 == 0:
            ax.get_ygridlines()[i].set_linewidth(3)
        else:
            ax.get_ygridlines()[i].set_linewidth(0)

    if no_numbers:
        # ax.grid(False)
        for i in range(len(confusion_matrix[:, 0])):
            for j in range(len(confusion_matrix[0, :])):
                value = round(confusion_matrix[i, j],1)
                value_ = str(value)
                color_index = value/np.max(confusion_matrix)
                if color_index > 0.35 or color_index > 0.65:
                    color_index = 1.0

                # if norm:
                #     value_ = str(int(confusion_matrix[i, j]))
                # for column, i in zip(confusion_matrix.T, range(np.shape(confusion_matrix)[1])):
                #     rgba[np.argmin(column), i, :] = 0, 1, 0, 0.2
                if np.argmin(confusion_matrix[:, j]) == i:
                    plt.annotate(value_, xy=(j - 0.25, i + 0.05), color=cmap_text(color_index), fontsize=14,
                                 fontweight='bold')
                    # ax = ax.add_subplot(111, aspect='equal')
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (j-0.5, i-0.5),  # (x,y)
                            0.95,  # width
                            0.95,  # height
                            color='#00ff99',
                            fill=False,
                            lw=1
                        )
                    )

                elif value < 10:
                    plt.annotate(value_, xy=(j - 0.25, i + 0.05), color=cmap_text(color_index), fontsize=14)
                elif value < 100:
                    plt.annotate(value_, xy=(j - 0.4, i + 0.05), color=cmap_text(color_index), fontsize=14)
                elif value < 1000:
                    plt.annotate(value_, xy=(j - 0.45, i + 0.05), color=cmap_text(color_index), fontsize=14)
                else:
                    plt.annotate(value_, xy=(j - 0.45, i + 0.05), color=cmap_text(color_index), fontsize=14)




    plt.draw()
    kwargs = dict(size=15, fontweight='medium')
    ax.set_yticklabels(labels_true, **kwargs)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(labels_pred, rotation=90, **kwargs)
    return ax

def prepare_confusion_matrix_plot(ax, confusion_matrix, labels_pred, labels_true, cmap,
                                  cmap_text, no_numbers, norm, N_Windows):
    if norm:
        for i in range(np.shape(confusion_matrix)[0]):
            confusion_matrix[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])
    labels_predz = []
    for i in range(len(labels_pred)):
        labels_predz.append("")
        labels_predz.append(labels_pred[i])

        labels_truez = []
    for i in range(len(labels_true)):
        labels_truez.append("")
        labels_truez.append(labels_true[i])

    labels_true = labels_truez
    labels_pred = labels_predz
    if norm:
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    elif N_Windows is not None:
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=N_Windows)
    else:
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    # plt.tight_layout()
    kwargs = dict(size=30, fontweight='bold')
    plt.ylabel('Model', **kwargs)
    plt.xlabel('Signal', **kwargs)
    ax.set_xticks(np.arange(-0.5, np.shape(confusion_matrix)[1], 0.5))
    ax.set_yticks(np.arange(-0.5, np.shape(confusion_matrix)[0], 0.5))

    for i in range(len(ax.get_xgridlines())):
        if i % 2 == 0:
            ax.get_xgridlines()[i].set_linewidth(3)
        else:
            ax.get_xgridlines()[i].set_linewidth(0)

    for i in range(len(ax.get_ygridlines())):
        if i % 2 == 0:
            ax.get_ygridlines()[i].set_linewidth(3)
        else:
            ax.get_ygridlines()[i].set_linewidth(0)

    if no_numbers:
        # ax.grid(False)
        for i in range(len(confusion_matrix[:,0])):
            for j in range(len(confusion_matrix[0,:])):
                value = round(confusion_matrix[i, j],2)
                value_ = str(value)
                color_index = value/np.max(confusion_matrix)
                if color_index>0.35 or color_index>0.65:
                    color_index = 1.0

                #     value_ = str(int(confusion_matrix[i, j]))
                if value < 10:
                    plt.annotate(value_, xy=(j - 0.35, i + 0.05), color=cmap_text(0.2), fontsize=10)
                elif value < 100:
                    plt.annotate(value_, xy=(j - 0.4, i + 0.05), color=cmap_text(0.2), fontsize=10)
                elif value < 1000:
                    plt.annotate(value_, xy=(j - 0.45, i + 0.05), color=cmap_text(0.2), fontsize=10)
                else:
                    plt.annotate(value_, xy=(j - 0.45, i + 0.05), color=cmap_text(0.2), fontsize=10)

    plt.draw()
    kwargs = dict(size=15, fontweight='medium')
    ax.set_yticklabels(labels_true, **kwargs)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(labels_pred, rotation=90, **kwargs)
    return ax

def load_loss(models, i, j):
    return np.load(SYNTH_DIRECTORY + "/LOSS_FOR_100_SYNTH_" + \
               models[i][0].name[:3] + "_" + models[j][0].name[:3]+".npz")["loss_tensor"]


def get_mean_and_std_matrices(all_models, source_ind, other_ind, loss_filename):
    loss_tensor = load_loss(all_models, source_ind, source_ind)

    loss_source = np.array([loss_tensor[i, i, :] for i in range(np.shape(loss_tensor)[0])])
    loss_type = np.array([loss_tensor[i, np.arange(np.shape(loss_tensor)[0]) != i] for i in np.arange(np.shape(loss_tensor)[0])])
    loss = [load_loss(all_models, source_ind, other_ind[0]), load_loss(all_models, source_ind, other_ind[1])]
    loss_other = [np.reshape(l, (np.shape(l)[0], np.shape(l)[1] * np.shape(l)[2])) for l in loss]
    loss_other = np.append(loss_other[0],
                           loss_other[1], axis=1)

    loss_type = np.reshape(loss_type,
                             (np.shape(loss_type)[0], np.shape(loss_type)[1] * np.shape(loss_type)[2]))

    confusion_mean = np.array([np.mean(loss_source, axis=1), np.mean(loss_type, axis=1), np.mean(loss_other, axis=1)])
    confusion_std = np.array([np.std(loss_source, axis=1), np.std(loss_type, axis=1), np.std(loss_other, axis=1)])

    return confusion_mean, confusion_std


def process_losses(i, loss_filename):
    for models in [all_models[i]]:
        for signal_batch, j in zip(all_signals, range(len(all_signals))):
            filename = SYNTH_DIRECTORY + loss_filename + models[0].name[:3] + "_" + all_models[j][0].name[:3]
            print(filename)
            if j == 1:
                overlap = 0.1
            else:
                overlap = 0.33

            N_Windows = 200000000000
            for signal in signal_batch:
                    if first_test_index != int(0.33 * len(signal)):
                        first_test_index = int(0.33 * len(signal))
                        signal_test = segment_signal(signal[first_test_index:], W, overlap)
                        N_Windows = len(signal_test[0]) if len(signal_test[0]) < N_Windows else N_Windows

            calculate_loss_tensor(filename, N_Windows, W, models, signal_batch, overlap=overlap)


def process_graphs(all_models, source_index, others_indexes, loss_filename, name):
    labels_pred = [mod.name for mod in all_models[source_index]]
    labels_true = ["Source "+name, "Other "+name, "Other Signals"]

    confusion_mean, confusion_std = get_mean_and_std_matrices(all_models, source_index, others_indexes, loss_filename)

    plot_emg_confusion_matrix(confusion_mean, labels_pred, labels_true, title='EMG', cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))

    plot_confusion_matrix(confusion_std, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Blues,
                          cmap_text=plt.cm.Blues_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))


if __name__ == "__main__":
    N_Windows = None
    W = 512
    signal_dim = 64
    hidden_dim = 256
    batch_size = 128
    window_size = 512
    fs = 250

    all_models = [db.resp_64_models, db.emg_64_models, db.ecg_64_models]
    all_signals = [np.load("../data/Fantasia_RESP_[64].npz")['x_train'].tolist(),
                   np.load("../data/FMH_[64].npz")['x_train'].tolist(),
                   np.load("../data/FANTASIA_ECG[64].npz")['x_train'].tolist()]
    first_test_index = 0

    # if N_Windows is None:
    #     N_Windows = 200000000000
    #     for signal in signals:
    #         if first_test_index != int(0.33 * len(signal)):
    #             first_test_index = int(0.33 * len(signal))
    #             signal_test = segment_signal(signal[first_test_index:], W, 0.10)
    #             N_Windows = len(signal_test[0]) if len(signal_test[0]) < N_Windows else N_Windows
    i = 1
    loss_filename = "/LOSS_FOR_100_SYNTH_"
    process_losses(i, loss_filename)
    loss_tensors = []
    # filename = SYNTH_DIRECTORY + "/LOSS_FOR_SYNTH_RESP_ENTROPY"
    process_graphs(all_models, 0, [1, 2], loss_filename, "RESP")
    process_graphs(all_models, 1, [0, 2], loss_filename, "EMG")
    process_graphs(all_models, 2, [0, 1], loss_filename, "ECG")

    # RESP
    # labels_pred = [mod.name for mod in all_models[0]]
    # labels_true = ["Source RESP", "Other RESP", "Other Signals"]
    #
    # confusion_mean, confusion_std = get_mean_and_std_matrices(all_models, 0, [1, 2], loss_filename)
    #
    # plot_emg_confusion_matrix(confusion_mean, labels_pred, labels_true, title='RESP', cmap=plt.cm.Reds,
    #                       cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))
    #
    # plot_confusion_matrix(confusion_std, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Blues,
    #                       cmap_text=plt.cm.Blues_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))

    # EMG
    labels_pred = [mod.name for mod in all_models[1]]
    labels_true = ["Source EMG", "Other EMGs", "Other Signals"]

    confusion_mean, confusion_std = get_mean_and_std_matrices(all_models, 1, [0, 2], loss_filename)

    plot_emg_confusion_matrix(confusion_mean, labels_pred, labels_true, title='EMG', cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))

    plot_confusion_matrix(confusion_std, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Blues,
                          cmap_text=plt.cm.Blues_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))
    #
    # # ECG
    labels_pred = [mod.name for mod in all_models[2]]
    labels_true = ["Source ECG", "Other ECGs", "Other Signals"]

    confusion_mean, confusion_std = get_mean_and_std_matrices(all_models, 2, [0, 1], loss_filename)

    plot_emg_confusion_matrix(confusion_mean, labels_pred, labels_true, title='ECG', cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))

    plot_confusion_matrix(confusion_std, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Blues,
                          cmap_text=plt.cm.Blues_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))

    # for loss_tensors
    # confusion_mean = np.mean(loss_tensor, axis=2)
    # plot_emg_confusion_matrix(confusion_mean, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Reds,
    #                       cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))
    #
    # confusion_std = np.std(loss_tensor, axis=2)
    # plot_confusion_matrix(confusion_std, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Blues,
    #                       cmap_text=plt.cm.Blues_r, no_numbers=True, norm=False, N_Windows=np.max(confusion_mean))
    #
    # resp_indexes = list(range(0, 20))
    # emg_indexes = list(range(20, 33))
    # ecg_indexes = list(range(33, 53))
    #
    # mean_loss_per_type = np.zeros((3, 3))
    # std_loss_per_type = np.zeros((3, 3))
    #
    # mean_loss_per_type[0, 0], mean_loss_per_type[0, 1], mean_loss_per_type[0, 2] = \
    #     np.mean(resp_indexes[:, resp_indexes]), \
    #     np.mean(resp_indexes[:, emg_indexes]), \
    #     np.mean(resp_indexes[:, ecg_indexes])
    # mean_loss_per_type[1, 0], mean_loss_per_type[1, 1], mean_loss_per_type[1, 2] = \
    #     np.mean(emg_indexes[:, resp_indexes]), \
    #     np.mean(emg_indexes[:, emg_indexes]), \
    #     np.mean(emg_indexes[:, ecg_indexes])
    # mean_loss_per_type[2, 0], mean_loss_per_type[2, 1], mean_loss_per_type[2, 2] = \
    #     np.mean(ecg_indexes[:, resp_indexes]), \
    #     np.mean(ecg_indexes[:, emg_indexes]), \
    #     np.mean(ecg_indexes[:, ecg_indexes])
    #
    # std_loss_per_type[0, 0], std_loss_per_type[0, 1], std_loss_per_type[0, 2] = \
    #     np.std(resp_indexes[:, resp_indexes]), \
    #     np.std(resp_indexes[:, emg_indexes]), \
    #     np.std(resp_indexes[:, ecg_indexes])
    # std_loss_per_type[1, 0], std_loss_per_type[1, 1], std_loss_per_type[1, 2] = \
    #     np.std(emg_indexes[:, resp_indexes]), \
    #     np.std(emg_indexes[:, emg_indexes]), \
    #     np.std(emg_indexes[:, ecg_indexes])
    # std_loss_per_type[2, 0], std_loss_per_type[2, 1], std_loss_per_type[2, 2] = \
    #     np.std(ecg_indexes[:, resp_indexes]), \
    #     np.std(ecg_indexes[:, emg_indexes]), \
    #     np.std(ecg_indexes[:, ecg_indexes])
    #
    # # for j in range(np.shape(mean_loss_per_type)[1]):
    # #     mean_loss_per_type[:, j] = mean_loss_per_type[:, j] / np.max(mean_loss_per_type[:, j])
    #
    #
    # print(mean_loss_per_type)
    # confusion_mean = np.mean(loss_tensor, axis=2)
    # plot_emg_confusion_matrix(mean_loss_per_type, ["ECG", "RESP", "EMG"], ["ECG", "RESP", "EMG"], title='Error per type',
    #                       cmap=plt.cm.Reds, cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=None)
    #
    #
    # plot_emg_confusion_matrix(std_loss_per_type, ["ECG", "RESP", "EMG"], ["ECG", "RESP", "EMG"], title='Error per type',
    #                       cmap=plt.cm.Reds, cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=None)