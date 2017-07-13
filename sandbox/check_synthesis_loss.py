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

        # if not os.path.isdir(os.path.dirname(filename + ".npz")):
        #     os.mkdir(os.path.dirname(filename + ".npz"))

        np.savez(filename + ".npz",
                 loss_tensor=loss_tensor,
                 signals_models=signals_models)

    return loss_tensor


def plot_confusion_matrix(confusion_matrix, labels_pred, labels_true, title='Confusion matrix' , cmap=plt.cm.Reds,
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
                if color_index > 0.35 or color_index > 0.65:
                    color_index = 1.0

                if norm:
                    value_ = str(int(confusion_matrix[i, j]*100)) + "%"
                if value < 10:
                    plt.annotate(value_, xy=(j - 0.1, i + 0.05), color=cmap_text(color_index), fontsize=20)
                elif value < 100:
                    plt.annotate(value_, xy=(j - 0.15, i + 0.05), color=cmap_text(color_index), fontsize=20)
                elif value < 1000:
                    plt.annotate(value_, xy=(j - 0.2, i + 0.05), color=cmap_text(color_index), fontsize=20)
                else:
                    plt.annotate(value_, xy=(j - 0.25, i + 0.05), color=cmap_text(color_index), fontsize=20)

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

                if norm:
                    value_ = str(int(confusion_matrix[i, j]*100)) + "%"
                if value < 10:
                    plt.annotate(value_, xy=(j - 0.1, i + 0.05), color=cmap_text(color_index), fontsize=20)
                elif value < 100:
                    plt.annotate(value_, xy=(j - 0.15, i + 0.05), color=cmap_text(color_index), fontsize=20)
                elif value < 1000:
                    plt.annotate(value_, xy=(j - 0.2, i + 0.05), color=cmap_text(color_index), fontsize=20)
                else:
                    plt.annotate(value_, xy=(j - 0.25, i + 0.05), color=cmap_text(color_index), fontsize=20)

    plt.draw()
    kwargs = dict(size=15, fontweight='medium')
    ax.set_yticklabels(labels_true, **kwargs)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(labels_pred, rotation=90, **kwargs)
    return ax

if __name__ == "__main__":
    N_Windows = None
    W = 256
    signal_dim = 64
    hidden_dim = 256
    batch_size = 128
    window_size = 256
    fs = 250

    models = db.ecg_64_models + db.resp_64_models + db.emg_64_models
    signals = np.load("../data/FANTASIA_ECG[64].npz")['x_train'].tolist() + \
              np.load("../data/Fantasia_RESP_[64].npz")['x_train'].tolist() + \
              np.load("../data/FMH_[64].npz")['x_train'].tolist()
    first_test_index = 0

    N_Windows = 200000000000
    for signal in signals:
        if first_test_index != int(0.33 * len(signal)):
            first_test_index = int(0.33 * len(signal))
            signal_test = segment_signal(signal[first_test_index:], W, 0.11)
            N_Windows = len(signal_test[0]) if len(signal_test[0]) < N_Windows else N_Windows

    loss_tensor = []
    filename = SYNTH_DIRECTORY + "/LOSS_FOR_SYNTH"
    # loss_tensor = calculate_loss_tensor(filename, N_Windows, W, models, signals, overlap=0.11)

    if len(loss_tensor) < 1:
        loss_tensor = np.load(filename + ".npz")["loss_tensor"]

    labels_true = labels_pred = ["ECG {0}".format(i) for i in range(1, 21)] + \
                                ["RESP {0}".format(i) for i in range(1, 21)] + \
                                ["EMG {0}".format(i) for i in (range(1, 15))]

    confusion_mean = np.mean(loss_tensor, axis=2)
    plot_confusion_matrix(confusion_mean, labels_pred, labels_true, title='Confusion matrix', cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=False, norm=False, N_Windows=None)

    confusion_std = np.std(loss_tensor, axis=2)
    plot_confusion_matrix(confusion_std, labels_pred, labels_true, title='Confusion matrix' , cmap=plt.cm.Blues,
                          cmap_text=plt.cm.Blues_r, no_numbers=False, norm=False, N_Windows=np.max(confusion_std))

    ecg_signals_index = list(range(1, 7)) + list(range(9, 19))
    resp_signals_index = list(range(20, 40))
    emg_signals_index = list(range(40, np.shape(loss_tensor)[1]))

    ecg_loss_tensor = loss_tensor[list(range(0, 19))] #/ np.max(loss_tensor[list(range(0, 19))], axis=1)
    resp_loss_tensor = loss_tensor[list(range(20, 37))] #/ np.max(loss_tensor[list(range(20, 37))], axis=1)
    emg_loss_tensor = loss_tensor[list(range(37, np.shape(loss_tensor)[0]))]
    #emg_loss_tensor = emg_loss_tensor / np.max(emg_loss_tensor, axis=1)

    mean_loss_per_type = np.zeros((3, 3))
    std_loss_per_type = np.zeros((3, 3))

    mean_loss_per_type[0,0], mean_loss_per_type[0,1], mean_loss_per_type[0,2] = \
        np.mean(ecg_loss_tensor[:, ecg_signals_index]), \
        np.mean(ecg_loss_tensor[:, resp_signals_index]), \
        np.mean(ecg_loss_tensor[:, emg_signals_index])
    mean_loss_per_type[1, 0], mean_loss_per_type[1, 1], mean_loss_per_type[1, 2] = \
        np.mean(resp_loss_tensor[:, ecg_signals_index]), \
        np.mean(resp_loss_tensor[:, resp_signals_index]), \
        np.mean(resp_loss_tensor[:, emg_signals_index])
    mean_loss_per_type[2, 0], mean_loss_per_type[2, 1], mean_loss_per_type[2, 2] = \
        np.mean(emg_loss_tensor[:, ecg_signals_index]), \
        np.mean(emg_loss_tensor[:, resp_signals_index]), \
        np.mean(emg_loss_tensor[:, emg_signals_index])

    # for j in range(np.shape(mean_loss_per_type)[1]):
    #     mean_loss_per_type[:, j] = mean_loss_per_type[:, j] / np.max(mean_loss_per_type[:, j])


    print(mean_loss_per_type)
    confusion_mean = np.mean(loss_tensor, axis=2)
    plot_confusion_matrix(mean_loss_per_type, ["ECG", "RESP", "EMG"], ["ECG", "RESP", "EMG"], title='Error per type',
                          cmap=plt.cm.Reds, cmap_text=plt.cm.Reds_r, no_numbers=True, norm=False, N_Windows=None)