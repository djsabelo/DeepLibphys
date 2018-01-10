import time

import numpy as np

import DeepLibphys.utils.functions.database as db
from novainstrumentation import smooth
import matplotlib.pyplot as plt
from DeepLibphys.utils.functions.signal2model import Signal2Model
import scipy.io as sio
import seaborn
from DeepLibphys.utils.functions.common import *
import DeepLibphys.models.LibphysMBGRU as GRU


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
    for i, signal in zip(indexes, signals[indexes]):
        name = 'ecg_' + str(i+1)

        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim, batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval, lower_error=3e-5, lower_learning_rate=1e-4, count_to_break_max=30)
        print("Compiling Model {0}".format(name))

        last_index = int(len(signal)*0.33)
        x_train, y_train = prepare_test_data([signal[22500:last_index]], signal2model, mean_tol=0.9, std_tol=0.5)


        # fig, ax = plt.subplots()
        # plt.subplots_adjust(bottom=0.2)
        # l, = plt.plot(x_train[0], lw=2)
        #
        # class BooleanSwitcher(object):
        #     indexes = []
        #     ind = 0
        #
        #     def yes(self, event):
        #         if self.ind < len(x_train):
        #             self.indexes.append(self.ind)
        #             self.ind += 1
        #         if self.ind < len(x_train):
        #             l.set_ydata(x_train[self.ind])
        #             plt.draw()
        #         else:
        #             self.crop()
        #             plt.close()
        #
        #     def no(self, event):
        #         self.ind += 1
        #         if self.ind < len(x_train):
        #             l.set_ydata(x_train[self.ind])
        #             plt.draw()
        #         else:
        #             self.crop()
        #             plt.close()
        #
        #     def crop(self):
        #         c = len(self.indexes) % 16
        #         self.indexes = self.indexes[:(len(self.indexes) - c)]
        # callback = BooleanSwitcher()
        # axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        # axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        # by = Button(axnext, 'Yes')
        # by.on_clicked(callback.yes)
        # bn = Button(axprev, 'No')
        # bn.on_clicked(callback.no)
        # plt.show()


        model = GRU.LibphysMBGRU(signal2model)
        # try:
        #
        #     # if i < 20:
        #     #     old_directory = "CLEAN_ECG_BIOMETRY[128.1024]"
        #     #     old_name = 'clean_ecg' + str(i+1)
        #     # else:
        #     old_directory = "BIOMETRY[256.1024]"
        #     old_name = name
        #
        #     old_tag= 'GRU_{0}[{1}.{2}.{3}.{4}.{5}]'. \
        #         format(old_name, signal_dim, hidden_dim, -1, -5, -5)
        #     model.load(old_tag, old_directory)
        # except:
        #     pass

        print("Initiating training... ")
        model.model_name = 'ecg_' + str(i+1)


        model.start_time = time.time()
        # returned = model.train_model(x_train[callback.indexes], y_train[callback.indexes], signal2model)
        model.load(model.get_file_tag(), signal_directory)
        returned = model.train_model(x_train, y_train, signal2model)
        if returned:
            model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))



model_info = db.ecg_1024_256_RAW[0]
signal_dim = model_info.Sd
hidden_dim = model_info.Hd
mini_batch_size = 16
batch_size = 256
window_size = 1024
save_interval = 10000
signal_directory = 'BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

indexes = np.arange(1, 41)

print("Loading signals...")
# signals = np.load("../data/signals_without_noise.npz")['signals_without_noise'][indexes]
# x_train, y_train = get_fantasia_dataset(signal_dim,  indexes, db.fantasia_ecgs[0].directory, peak_into_data=False)

# np.savez("../data/processed/FANTASIA_ECG[256].npz", x_train=x_train, y_train=y_train)
x_train, y_train = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train'], \
                   np.load("../data/processed/FANTASIA_ECG[256].npz")['y_train']
# np.savez("../data/processed/backup_FANTASIA_ECG[256].npz", x_train=x_train_1, y_train=y_train_1)
# x_train_1[14] = x_train[14]
# y_train_1[14] = y_train[14]
# np.savez("../data/processed/FANTASIA_ECG[256].npz", x_train=x_train_1, y_train=y_train_1)
# for i, x in zip(x_train, x_train_1):
#     plt.figure()
#     plt.plot(i)
#     plt.plot(x)
# plt.show()
# for x in x_train:
#     plt.plot(x)
#     plt.show()
# print(xxasxa)
# indexes = np.array([8, 9])
# indexes = np.array([18, 19, 26])
# indexes = np.array([27, 28])
indexes = np.array([32])
# plt.plot(x_train[14])
# plt.show()
# indexes = np.array([38, 39])

# indexes = np.arange(20, 30)
# indexes = np.arange(30, 40)
train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, signal_directory, indexes, x_train, save_interval,
               signal_dim)
