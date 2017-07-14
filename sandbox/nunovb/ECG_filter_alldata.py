from DeepLibphys.utils.functions.common import quantize_signal, get_fantasia_full_paths, remove_noise, segment_signal, ModelType
from DeepLibphys.utils.functions.signal2model import Signal2Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from DeepLibphys.models import LibphysMBGRU, LibphysSGDGRU
import time

#print(get_fantasia_full_paths()[0])
def train_block(signals_train,signals_test, signal2model, signal_indexes=None, n_for_each=12, overlap=0.33,random_training=True,
                start_index=0, track_loss=False, loss_interval=1, train_ratio=1):
    """
    This method embraces several datasets (or one) according to a number of records for each

    :param signals: - list - a list containing two int vectors:
                                signal - input vector X, used for the input;

    :param signal2model: - Signal2Model object - object containing the information about the model, for more info
                            check Biosignals.utils.functions.signal2model

    :param signal_indexes: - list - a list containing the indexes of the "signals" variable to be trained.
                                    If None is given, all signals will be used.

    :param n_for_each: - int - number of windows from each signal to be inserted in the model training

    :param overlap: - float - value in the interval [0,1] that corresponds to the overlapping ratio of windows

    :param random_training: - boolean - value that if True random windows will be inserted in the training

    :param start_index: - int - value from which the windows will be selected

    :param track_loss: - boolean - value to plot loss as the model is trained

    :return: trained model
    """

    if signal_indexes is None:
        signal_indexes = range(len(signals_train))
    print("indexes:",signal_indexes)
    #self.save(signal2model.signal_directory, self.get_file_tag(-1, -1))

    x_train = []
    y_train = []
    for i in signal_indexes:

        # Creation of the Time Windows from the dataset
        if n_for_each == 1:
            if len(x_train) == 0:
                x_train = signals_train[i][:signal2model.window_size]
                y_train = signals_test[i][1:signal2model.window_size + 1] # for next signal without noise, [1:window_size + 1]
            else:
                x_train = np.vstack((x_train, signals_train[i][:signal2model.window_size]))
                y_train = np.vstack((y_train, signals_test[i][1:signal2model.window_size + 1]))
        else:
            X_windows, y_end_values, n_windows, last_index = segment_signal(signals_train[i][:-1], signal2model.window_size,
                                                                            overlap=overlap, start_index=start_index)
            Y_windows, y_end_values, n_windows, last_index = segment_signal(signals_test[i][1:], signal2model.window_size,
                                                                            overlap=overlap, start_index=start_index)

            n_for_each = n_for_each if n_for_each < np.shape(X_windows)[0] else np.shape(X_windows)[0]
            n_for_each = n_for_each if n_for_each % signal2model.mini_batch_size == 0 \
                else signal2model.mini_batch_size * int(n_for_each / signal2model.mini_batch_size)

            last_training_index = int(n_windows * train_ratio)
            # List of the windows to be inserted in the dataset
            if random_training:
                window_indexes = np.random.permutation(last_training_index)  # randomly select windows
            else:
                window_indexes = list(range((n_windows)))  # first windows are selected

            # Insertion of the windows of this signal in the general dataset
            if len(x_train) == 0:
                # First is for train data
                x_train = X_windows[window_indexes[0:n_for_each], :]
                y_train = Y_windows[window_indexes[0:n_for_each], :]
                print("x_train shape:", x_train.shape)

                # # The rest is for test data
                # x_test = X_windows[last_training_index:, :]
                # y_test = Y_windows[last_training_index:, :]
            else:
                x_train = np.append(x_train, X_windows[window_indexes[0:n_for_each], :], axis=0)
                y_train = np.append(y_train, Y_windows[window_indexes[0:n_for_each], :], axis=0)
                # x_test = np.append(x_train, X_windows[window_indexes[n_for_each:], :], axis=0)
                # y_test = np.append(x_train, Y_windows[window_indexes[n_for_each:], :], axis=0)

                # Save test data
                # self.save_test_data(signal2model.signal_directory, [x_test, y_test])

    # Start time recording


    # Start training model
    model = LibphysMBGRU.LibphysMBGRU(signal2model) #signal2model, ModelType.CROSS_MBSGD, params)) -> for LibphysGRU
    t1 = time.time()
    model.start_time = time.time()
    returned = model.train_model(x_train, y_train, signal2model, track_loss, loss_interval)

    print("Dataset trained in: ~%d seconds" % int(time.time() - t1))

    # Model last training is then saved
    if returned:
        model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))
        return True
    else:
        return False

def generate_predicted_signal(model, N, starting_signal, window_seen_by_GRU_size):
    # We start the sentence with the start token
    new_signal = []
    # Repeat until we get an end token
    print('Starting model generation')
    percent = 0
    plt.ion()
    for i in range(2, N):
        if int(i * 100 / N) % 5 == 0:
            print('.', end='')
        elif int(i * 100 / N) % 20 == 0:
            percent += 0.2
            print('{0}%'.format(percent))

        test_signal = starting_signal[:i]
        sample, output = generate_online_predicted_signal(model, test_signal, window_seen_by_GRU_size)
        new_signal.append(sample)

        plt.figure(0)
        plt.clf()
        plt.plot(new_signal)
        plt.pause(0.05)
        # plt.figure(1)
        # plt.clf()
        # plt.plot(output)
        # plt.pause(0.1)

    return new_signal[1:]

def generate_online_predicted_signal(model, starting_signal, window_seen_by_GRU_size, uncertainty=0.05):
    new_signal = starting_signal
    next_sample = None
    output = 0
    next_sample_probs = np.empty_like(new_signal)
    try:
        # if new_signal.shape[0] <= window_seen_by_GRU_size:
        #     signal = new_signal
        # else:
        #     signal = new_signal[-window_seen_by_GRU_size:]
        signal = new_signal
        next_sample_probs = np.empty_like(signal)
        [output] = model.predict(signal)
        print(output)
        next_sample_probs = np.asarray(output, dtype=float)

        next_sample_probs = next_sample_probs[-1] / np.sum(next_sample_probs[-1])
        next_sample = np.random.choice(model.signal_dim, p=next_sample_probs)

    except:
        print("exception: " + np.sum(np.asarray(next_sample_probs[-1]), dtype=float))
        next_sample = 0
    # plt.plot(np.arange(len(next_sample_probs)), next_sample_probs, 'k-', next_sample, xxx[next_sample], 'ro')
    # plt.plot(np.arange(len(xxx)), xxx, 'g-')
    # print(next_sample)
    # plt.show()
    return next_sample, output[-1]

# Load Signals
original_signals = np.array([loadmat(get_fantasia_full_paths()[i])['val'][0][:30000] for i in range(len(get_fantasia_full_paths()))])
print(original_signals.shape)

# Apply filter
filtered_signals = np.array([remove_noise(original_signals[i]) for i in range(original_signals.shape[0])])

# Reduce the dimensionality
original_signals = np.array([quantize_signal(original_signals[i], 256) for i in range(original_signals.shape[0])])

filtered_signals = np.array([quantize_signal(filtered_signals[i], 256) for i in range(filtered_signals.shape[0])])

print("Quantization Finished")

save_directory = "bento"
for i in range(original_signals.shape[0]):
    sig_model = Signal2Model("GRU_Filter_all_"+str(i), save_directory, signal_dim=256, mini_batch_size=16, window_size=512)
    train_block([original_signals[i]], [filtered_signals[i]], n_for_each=128, signal2model=sig_model)

# test_signals = np.array([loadmat(get_fantasia_full_paths()[i])['val'][0][30000:60000] for i in range(len(get_fantasia_full_paths()))])
# model = LibphysSGDGRU.LibphysSGDGRU(sig_model)
# model.load(model.get_file_tag(-5, -5), sig_model.signal_directory)
# predicted = generate_predicted_signal(sig_model, starting_signal=test_signals[0],window_seen_by_GRU_size=1024)
# Predict? Do we have to do save and load?

# plt.subplot(311)
# plt.plot(original_signals[0])
# plt.subplot(312)
# plt.plot(filtered_signals[0])
# plt.subplot(313)
# plt.plot(predicted)
# plt.show()


