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
from hmmlearn import hmm
from sklearn.externals import joblib
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


def train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, signals,save_interval,signal_dim):
    for i, signal in zip(indexes, signals):
        name = 'ecg_' + str(i)
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim, batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval)
        print("Compiling Model {0}".format(name))
        model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2model)
        print("Initiating training... ")
        model.train(signal, signal2model)

signal_dim = 64
hidden_dim = 256
mini_batch_size = 16
batch_size = 128
window_size = 512
save_interval = 1000
signal_directory = 'BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

indexes = np.arange(1, 21)

print("Loading signals...")
# signals = np.load("../data/signals_without_noise.npz")['signals_without_noise'][indexes]
# x_train, y_train = get_fantasia_dataset(signal_dim, indexes, db.fantasia_ecgs[0].directory, peak_into_data=False)
# np.savez("../data/FANTASIA_ECG[64].npz", x_train=x_train, y_train=y_train)
x_train, y_train = np.load("../data/FANTASIA_ECG[64].npz")['x_train'], np.load("../data/FANTASIA_ECG[64].npz")['y_train']

# joblib.load("filename.pkl")

i = 0
for signal in x_train:
    i += 1
    print(i)
    X = signal[11000:int(11000+(512*128*0.33))].reshape(-1, 1)
    model = hmm.GaussianHMM(signal_dim, n_iter=10000, tol=0.005)
    model.fit(X)
    plt.plot(model.predict(X))
    plt.show()
    joblib.dump(model, "../data/hmm_ecg_"+str(i)+".pkl")

# N = 100
# for i, signal in zip(range(1, len(x_train)+1), x_train):
#     # try:
#         X = [np.random.randint(0, signal_dim-1)]
#         X_ = []
#         for n in range(N):
#             model = joblib.load("../data/hmm_ecg_"+str(i)+".pkl")
#             x = np.array(X).reshape(-1, 1)
#             y = model.predict(x)
#             X += [y[-1]]
#             print(X)
#         plt.plot(X)
#         plt.show()

N = 100
for i, signal in zip(range(1, len(x_train) + 1), x_train):
    # try:
    X = [np.random.randint(0, signal_dim - 1)]
    X_ = []

    model = joblib.load("../data/hmm_ecg_" + str(i) + ".pkl")
    X = signal[11000:int(11000 + (512 * 128 * 0.33))].reshape(-1, 1)
    y = model.predict(X)
    print(X[1:] - y)
    plt.plot(X[1:])
    plt.plot(y)
    plt.show()

    # except:
    #     print("Error in "+str(i))