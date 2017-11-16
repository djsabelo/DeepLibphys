import time

import numpy as np

import DeepLibphys
import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from novainstrumentation import smooth
from DeepLibphys.utils.functions.common import *
import matplotlib.pyplot as plt
from DeepLibphys.utils.functions.signal2model import Signal2Model
import scipy.io as sio
import seaborn


def process_and_save_fantasia(plot=False, signal_dim=64):
    """
    insert noise into fantasia dataset
    
    :param plot: 
    :param signal_dim: 
    :return: 
    """

    signals_without_noise = []
    signals_noise = []
    SNR = []
    full_paths = get_fantasia_full_paths(db.fantasia_ecgs[0].directory, list(range(1,21)))
    for file_path in full_paths:
        # try:
        print("Pre-processing signal - " + file_path)
        signal = sio.loadmat(file_path)['val'][0][:1256]
        # plt.plot(signal)
        processed = process_dnn_signal(signal, signal_dim)
        signals_without_noise.append(processed)
        if plot:
            plt.plot(processed)
            plt.show()
            plt.plot(signal[1000:5000])
            fig, ax = plt.subplots()
            major_ticks = np.arange(0, 64)
            ax.set_yticks(major_ticks)
            plt.ylim([0, 15])
            plt.xlim([0, 140])
            ax.grid(True, which='both')
            plt.minorticks_on
            # ax.grid(which="minor", color='k')
            ax.set_ylabel('Class - k')
            ax.set_xlabel('Sample - n')
            plt.plot(signalx, label="Smoothed Signal", alpha=0.4)
            plt.plot(processed, label="Discretized Signal")
            # ticklines = ax.get_xticklines() + ax.get_yticklines()
            gridlines = ax.get_ygridlines()  # + ax.get_ygridlines()
            ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

            for line in gridlines:
                line.set_color('k')
                line.set_linestyle('-')
                line.set_linewidth(1)
                line.set_alpha(0.2)

            for label in ticklabels:
                label.set_color('r')
                label.set_fontsize('medium')
            plt.legend()

            plt.show()

        signalx = smooth(remove_moving_std(remove_moving_avg((signal - np.mean(signal)) / np.std(signal))))

    print("Saving signals...")
    np.savez("signals_without_noise.npz", signals_without_noise=signals_without_noise)


def train_FMH(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, signals,save_interval,signal_dim):
    for i, signal in zip(indexes, signals[indexes]):
        name = 'emg_' + str(i+1)
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                    batch_size=batch_size, mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval, tolerance=1e-12)

        returned = False
        while not returned:
            print("Compiling Model {0}".format(name))
            model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
            print("Initiating training... ")
            returned = model.train(signal, signal2model, overlap=0.05)

def train_other_FMH(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, signals,
              save_interval, signal_dim, accs):
    for i, signal in zip(indexes, signals[indexes]):
        name = 'emg_' + str(i + 1)
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                    batch_size=batch_size, mini_batch_size=mini_batch_size,
                                    window_size=window_size,
                                    save_interval=save_interval, tolerance=1e-12)

        returned = False
        acc = accs[i]
        emg = signal
        while not returned:
            print("Compiling Model {0}".format(name))
            model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
            print("Initiating training... ")
            returned = train(acc, emg, signal2model.batch_size, model, signal2model)

def train(acc, emg, batch_size, model, signal2model):
        indexes_for_cut = np.where(np.logical_and(np.diff(acc[:-1])*np.diff(acc[1:])<0,
                                                  acc[1:-1] > 0.8*np.mean(np.abs(acc))))[0][2:-100]

        indexes = [indexes_for_cut[i] for i in range(0, len(indexes_for_cut), 2)]
        indexes_for_cut = indexes
        max_size = int(np.mean(np.diff(indexes_for_cut)))
        print(max_size)

        X_windows = np.zeros((batch_size, max_size))
        Y_windows = np.zeros((batch_size, max_size))
        indexes_for_cut_train = indexes_for_cut[:int(len(indexes_for_cut)*0.33)]
        for i, index in zip(range(batch_size), np.random.permutation(indexes_for_cut_train)[:batch_size]):
            X_windows[i] = emg[index:index+max_size]
            Y_windows[i] = emg[1:][index:index+max_size]
            # plt.plot(X_windows[i])
            # plt.show()

        model.start_time = time.time()
        t1 = time.time()

        # Start training model
        returned = model.train_model(X_windows, Y_windows, signal2model)
        print("Dataset trained in: ~%d seconds" % int(time.time() - t1))

        # Model last training is then saved
        if returned:
            model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))
            return True
        else:
            return False



# 1,3,5,7,9,10,11
signal_dim = 64
hidden_dim = 512
mini_batch_size = 4
batch_size = 32
window_size = 512
save_interval = 1000
signal_directory = 'SYNTHESIS[{0}.{1}]'.format(batch_size, window_size)

indexes = np.array([11])#np.arange(7, 8)
# accz, emgs = get_fmh_emg_datset(20000, row=9, example_index_array=indexes)
# np.savez('accz.pnz', accz=accz)
# accz = np.load('accz.npz')['accz']
# emgs = np.load("../data/FMH_[64].npz")['x_train']
# accs = [acc - np.mean(acc) for acc in accz]
# # accz = accz-np.min(accz)
# # accz = accz*64/np.max(accz)
# train_other_FMH(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, emgs,
#               save_interval, signal_dim, accs)

# train_FMH(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, x_train, save_interval,
#            signal_dim)
N_Windows = None
W = 512
first_test_index = 0
signals = np.load("../data/FANTASIA_ECG[64].npz")['x_train'].tolist()
if N_Windows is None:
    N_Windows = 200000000000
    for signal in signals:
        if first_test_index != int(0.33 * len(signal)):
            first_test_index = int(0.33 * len(signal))
            signal_test = segment_signal(signal[first_test_index:], W, 0.33)
            N_Windows = len(signal_test[0]) if len(signal_test[0]) < N_Windows else N_Windows

print("Loading signals...")
# x_train, y_train = get_fmh_emg_datset(signal_dim, dataset_dir="EMG_Lower", row=0)
# np.savez("../data/FMH_[64].npz", x_train=x_train, y_train=y_train)
x_train, y_train = np.load("../data/FMH_[64].npz")['x_train'], np.load("../data/FMH_[64].npz")['y_train']
# print(len(x_train))


train_FMH(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, x_train, save_interval,
               signal_dim)
