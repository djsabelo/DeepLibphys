import os

import matplotlib as mpl
import matplotlib.colors as col
import matplotlib.patheffects as pte
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy as scp
import scipy.signal as sig
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from novainstrumentation.smooth import smooth
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
import sys

import DeepLibphys.models.libphys_GRU as GRU

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/'
TRAINED_DATA_DIRECTORY = '../data/trained/'
COLORMAP_DIRECTORY = '../data/biosig_colormap/'
RAW_SIGNAL_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/'
FANTASIA_ECG = 'Fantasia/ECG/mat/'
FANTASIA_RESP = 'Fantasia/RESP/mat/'

class ModelType:
    MINI_BATCH, SGD, CROSS_SGD, CROSS_MBSGD = range(4)

"""#####################################################################################################################
########################################################################################################################
###################                           DATASET PRE-PROCESS                         ##############################
########################################################################################################################
#####################################################################################################################"""


def acquire_and_process_signals(full_paths, signal_dim, decimate=None, peak_into_data=False, val='val', row=0,
                                smooth_window=10, val2=None, regression=False):
    signals = []
    count = 0

    for file_path in full_paths:
        try:
            signal = []
            extension = file_path[-3:]
            if peak_into_data:
                plt.figure(file_path[-12:])

            if extension == 'txt':
                signal = np.loadtxt(file_path)[:, row]       # load signals from txt dataset
            elif extension == 'mat':
                xx = sio.loadmat(file_path)
                if val2 is not None:
                    signal = sio.loadmat(file_path)[val][val2][0][0][:, row]  # load signals from matlab dataset from EEG records
                else:
                    signal = xx[val][row]    # load signals from matlab dataset
            elif extension == 'npz':
                signal = np.load(file_path)[val]  # load files from npzfile

            if (decimate is not None and isinstance(decimate, int)) and not isinstance(decimate, bool):
                signal = sig.decimate(signal, decimate)

            signals.append(process_signal(remove_noise(signal,smooth_window=smooth_window),
                                          signal_dim, peak_into_data, decimate, regression=regression))  # insert signals into an array of signals

            if peak_into_data is not False:
                start_index = np.random.random_integers(0, len(signal) - peak_into_data)
                plt.plot(signal[start_index:start_index + peak_into_data])
                plt.show()

            count += 1
        except TypeError as e:
            print(e)
        except:
            print("An expection has occured in " + file_path)

    if peak_into_data is not False:
        plt.show()
        plt.figure()
        count = 0
        # for signal in signals:
        #     start_index = np.random.random_integers(0, len(signal) - peak_into_data)
        #     plt.plot(signal[start_index:start_index + peak_into_data], label=str(count))
        #     count += 1
        plt.legend()
        plt.show()

    if len(full_paths) == 1:
        return signals[0]

    return signals

def remove_noise(signal, moving_avg_window=60, smooth_window=10):
    signal = smooth(remove_moving_avg(signal, moving_avg_window), smooth_window) # smooth the signal
    return signal

def process_signal(signal, interval_size, peak_into_data, decimate, regression):
    if (decimate is not None and isinstance(decimate, int)) and not isinstance(decimate, bool):
        signal = signal[100000:400000]

    # if peak_into_data is not False:
    #     start_index = np.random.random_integers(0, len(signal) - peak_into_data)
    #     plt.plot(signal[start_index:start_index + peak_into_data]-np.mean(signal[start_index:start_index + peak_into_data]), color='#660033')
    #     plt.show()

    if regression:
        signal -= np.mean(signal)
        return signal / np.max(abs(signal))  # divided by the max of the signal [-1;1]
    else:
        signal = signal.astype(int)  # transformed into an int signal representation
        signal -= np.min(signal)  # removed the minimum value
        signal = signal / int(np.max(signal) - np.min(signal))  # made a discrete representation of the signal
        signal *= (interval_size - 1)                           # with "interval_size" steps (min = 0, max = interval_size-1))
        return signal.astype(int)                               # insert signal into an array of signals


def process_dnn_signal(signal, interval_size, window_smooth=10, window_rmavg=60):
    """

    :param signal:
    :param interval_size:
    :param window_smooth:
    :param window_rmavg:
    :return:
    """
    # plt.plot(signal[1000:2000])

    signal = (signal - np.mean(signal, axis=0)) / np.std(signal, axis=0)
    # plt.figure(1)
    # plt.plot(signal[1000:111000])
    # # plt.show()

    signal = remove_moving_avg(signal, window_rmavg)
    # plt.figure(2)
    # plt.plot(signal[1000:111000])
    # # plt.show()

    signals = remove_moving_std(signal)

    if len(np.shape(signal)) > 1:
        print("processing signals")
        signalx = np.array([smooth(signal_, window_smooth) for signal_ in signals])
        # plt.plot(signalx[0][1000:2000])
        # plt.show()
        return np.array([quantize_signal(signal_, interval_size) for signal_ in signalx])
    else:
        signal = smooth(signal, window_smooth)
        # plt.figure()
        # plt.plot(signal[1000:2000])
        # plt.show()
        # ecg = quantize_signal(signal, interval_size)

        return quantize_signal(signal, interval_size)

def process_cnn_signal(signal, window_smooth=10, window_rmavg=60):
    """

    :param signal:
    :param interval_size:
    :param window_smooth:
    :param window_rmavg:
    :return:
    """
    # plt.plot(signal[1000:2000])

    signal = (signal - np.mean(signal, axis=0)) / np.std(signal, axis=0)
    # plt.figure(1)
    # plt.plot(signal[1000:111000])
    # # plt.show()

    signal = remove_moving_avg(signal, window_rmavg)
    # plt.figure(2)
    # plt.plot(signal[1000:111000])
    # # plt.show()

    signals = remove_moving_std(signal)

    return smooth(signal, window_smooth)
    # if len(np.shape(signal)) > 1:
    #     print("processing signals")
    #     return np.array([smooth(signal_, window_smooth) for signal_ in signals])
    # else:
    #     return smooth(signal, window_smooth)


def quantize_signal(signal, interval_size, confidence = 0.001):
    """
    Transform the signal's continuous values into discrete ones, given the interval size, in order to reduce its
    dimensionality
    :param signal: numpy array of floats
        Array of numbers that symbolize the signal that must be quantized to serve as input to the Deep Neural Networks.
    :param interval_size: int
        Number of discrete values attained by the signal
    :param confidence: float
        Percentage of the accepted observations from the histogram. The edge values which fall below the confidence
        are removed.
    :return: numpy array of ints
    """
    n, bins, patches = plt.hist(signal, 10000)
    distribution_sum = np.cumsum(n)
    MIN = bins[np.where(distribution_sum <= confidence * np.sum(n))[0][-1]]
    MAX = bins[np.where(distribution_sum >= (1-confidence) * np.sum(n))[0][0]]
    signal[signal >= MAX] = MAX
    signal[signal <= MIN] = MIN

    signal -= np.min(signal)                            # removed the minimum value
    signal = signal/np.max(signal)                      # made a discrete representation of the signal
    signal *= (interval_size - 1)                       # with "interval_size" steps (min = 0, max = interval_size-1))
    # signal = np.around(signal)
    plt.clf()
    return np.around(signal, 0).astype(np.int)            # insert signal into an array of signals


def process_web_signal(signal, interval_size, smooth_window, peak_into_data, decimate=None, window=1, smooth_type='hanning'):
                   # smooth the signal
    if (decimate is not None and isinstance(decimate, int)) and not isinstance(decimate, bool):
        signal = sig.decimate(signal, decimate)

    signal = smooth(signal, smooth_window, window=smooth_type)
    signal = remove_moving_avg(signal, window)
    if peak_into_data is not False:
        start_index = np.random.random_integers(0, len(signal) - peak_into_data)
        plt.plot(signal[start_index:start_index + peak_into_data]-np.mean(signal[start_index:start_index + peak_into_data]), color='#660033')
        plt.show()

    signal = signal.astype(int)  # transformed into an int signal representation
    signal -= np.min(signal)  # removed the minimum value
    signal = signal / int(np.max(signal) - np.min(signal))  # made a discrete representation of the signal
    signal *= (interval_size - 1)                           # with "interval_size" steps (min = 0, max = interval_size-1))
    return signal.astype(int)                               # insert signal into an array of signals

def remove_moving_avg(signal, window_size=60):
    signalx = np.zeros_like(signal)
    for i in range(len(signal)):
        n = [int(i - window_size/2), int(window_size/2 + i)]

        # if n[0] < 0:
        #     n[0] = 0
        if n[1] > len(signal):
            n[1] -= len(signal)

        if len(signal[n[0]:n[1]]) > 0:
            signalx[n[0]:n[1]] = (signal[n[0]:n[1]] - np.mean(signal[n[0]:n[1]], axis=0))

    return signalx


def remove_moving_std(signal, window_size=2000):
    signalx = np.zeros_like(signal)
    for i in range(len(signal)):
        n = [int(i - window_size / 2), int(window_size / 2 + i)]

        # if n[0] < 0:
        #     n[0] = 0
        if n[1] > len(signal):
            n[1] -= len(signal)

        if len(signal[n[0]:n[1]]) > 0:
            signalx[n[0]:n[1]] = signal[n[0]:n[1]] / np.std(signal[n[0]:n[1]], axis=0)

    return signalx

"""#####################################################################################################################
########################################################################################################################
######################                           DATASET LOADS                         #################################
########################################################################################################################
#####################################################################################################################"""


def get_fantasia_dataset(signal_dim, example_index_array, dataset_dir=FANTASIA_ECG, peak_into_data=False):
    full_paths = get_fantasia_full_paths(dataset_dir, example_index_array)
    signals = []
    for file_path in full_paths:
        signal = sio.loadmat(file_path)['val'][0]
        # signal = (signal - np.mean(signal)) / np.std(signal)
        if dataset_dir.count(FANTASIA_RESP)>0:
            signals.append(process_dnn_signal(signal, signal_dim, 100, 1000))
        else:
            signals.append(process_dnn_signal(signal, signal_dim))


        if peak_into_data:
            if peak_into_data == True:
                peak_into_data = 1000
            plt.figure(1)
            plt.plot(signal[1000:1000+peak_into_data])
            plt.figure(2)
            plt.plot(signals[-1][1000:1000+peak_into_data])
            plt.show()

    y_train = np.zeros(len(signals)).tolist()
    print("Signal acquired")
    for i in range(len(signals)):
        y_train[i] = signals[i][1:] + [0]

    return signals, y_train

def get_fantasia_full_paths(dataset_dir=FANTASIA_ECG, example_index_array=None):
    if example_index_array is None:
        example_index_array = np.arange(1,21)
    else:
        example_index_array = np.asarray(example_index_array)

    full_paths = np.zeros(len(example_index_array)).tolist()
    i = 0
    for example in example_index_array:
        file_name = ''
        if example <= 9:
            file_name = 'f1o0' + str(example) + 'm.mat'
        elif example == 10:
            file_name = 'f1o10m.mat'
        elif example <= 19:
            file_name = 'f1y0' + str(example-10) + 'm.mat'
        elif example == 20:
            file_name = 'f2y10m.mat'

        full_paths[i] = RAW_SIGNAL_DIRECTORY + dataset_dir + file_name
        i += 1

    return full_paths


def get_cyb_dataset_files(signal_dim, val='val', row=0, dataset_dir=FANTASIA_ECG, peak_into_data=False):
    full_paths = os.listdir(dataset_dir)
    train_signals = []
    test_signals = []
    fs = 1000
    train_names = []
    train_dates = []
    test_dates = []
    test_names = []
    for file_path in full_paths:
        if file_path[-3:] == "txt":
            file = dataset_dir+'/'+file_path
            print("Processing file: {0}".format(file))
            try:
                filename = file_path.split('/')[-1].split('.')[0]
                name = filename[9:-6]
                date = filename[:8]


                signal = np.loadtxt(file)
                N = len(signal)
                time = len(signal)/fs
                # iter_ = scp.interpolate.interp1d(np.arange(0, time, 1/fs), signal)
                # t = np.arange(0, time-1/500, 1/500)
                # signal = iter_(t)
                signal = sig.decimate(signal, 4)
                signal = process_dnn_signal(signal, signal_dim)
                if train_names.count(name) == 0:
                    train_signals.append(signal)
                    train_names.append(name)
                    train_dates.append(date)
                else:
                    test_signals.append(signal)
                    test_names.append(name)
                    test_dates.append(date)
                #
                #         if peak_into_data:
                if peak_into_data == True:
                    peak_into_data = 1000
                    plt.plot(signal[1000:1000+peak_into_data])
                    plt.show()
                print("Time: {0} s; Length 1: {1};Length 2: {2}".format(time, N, len(signal)))
            except ValueError:
                print("Error")
                pass

    return train_dates, train_names, train_signals, test_dates, test_names, test_signals


def get_mit_dataset_files(signal_dim, val='val', row=0, dataset_dir=FANTASIA_ECG, peak_into_data=False):
    full_paths = os.listdir(dataset_dir)
    signals = []
    fs = 360
    for file_path in full_paths:
        if file_path[-3:] == "mat":
            file = dataset_dir+'/'+file_path
            print("Processing file: {0}".format(file))
            try:
                signal = sio.loadmat(file)[val][row]
                N = len(signal)
                time = len(signal)/fs
                iter_ = scp.interpolate.interp1d(np.arange(0, time, 1/fs), signal)
                t = np.arange(0, time-1/500, 1/500)
                signal = iter_(t)
                signal = sig.decimate(signal, 2)
                signal = process_dnn_signal(signal, signal_dim)
                signals.append(signal)
                #
                #         if peak_into_data:
                if peak_into_data == True:
                    peak_into_data = 1000
                    plt.plot(signal[1000:1000+peak_into_data])
                    plt.show()
                print("Time: {0} s; Length 1: {1};Length 2: {2}".format(time, N, len(signal)))
            except ValueError:
                print("Error")
                pass

    return signals


# def get_fantasia_noisy_data(signal_dim, example_index_array, noisy_index, dataset_dir, peak_into_data, regression, smooth_window=10):
#     full_paths = np.zeros(len(example_index_array)).tolist()
#     example_index_array = np.asarray(example_index_array)
#     i = 0
#     for example in example_index_array:
#         file_name = ''
#         if example <= 9:
#             file_name = 'f1o0' + str(example) + 'm'
#         elif example == 10:
#             file_name = 'f1o10m'
#         elif example <= 19:
#             file_name = 'f1y0' + str(example-10) + 'm'
#         elif example == 20:
#             file_name = 'f1y10m'
#
#         full_paths[i] = RAW_SIGNAL_DIRECTORY + dataset_dir + file_name + '/' + 'value'+str(noisy_index)+'_'+file_name+'.npz'
#         i += 1
#     signals = acquire_and_process_signals(full_paths, signal_dim, peak_into_data=peak_into_data, smooth_window=smooth_window, regression=regression, val='signal')
#
#     if len(signals) > 50:
#         signals = [signals]
#
#     y_train = np.zeros(len(signals)).tolist()
#     print("Signal acquired")
#     for i in range(len(signals)):
#         y_train[i] = signals[i][1:] + [0]
#
#     return signals, y_train

def get_day_dataset(signal_dim, dataset_dir, file_name='ah_r-r', peak_into_data=False, smooth_window=10, regression=False, val='interp'):
    npzfile = np.load(RAW_SIGNAL_DIRECTORY + dataset_dir + file_name + '.npz', encoding='latin1')
    loaded_signals = npzfile[val]

    loaded_signals = [[process_signal(signal, signal_dim, peak_into_data, False, regression=regression)
                     for signal in signal_group]
                     for signal_group in loaded_signals]

    for signal in loaded_signals[0]:
        if peak_into_data is not False:
            start_index = np.random.random_integers(0, len(signal) - peak_into_data)
            plt.plot(signal[start_index:start_index + peak_into_data])
            plt.show()

    print("Signals acquired")
    return loaded_signals


def get_signals(signal_dim, dataset_dir, peak_into_data=False, decimate=None, val='val', row=0, smooth_window=10, val2=None, regression=False):
    full_paths = []
    for file in os.listdir(RAW_SIGNAL_DIRECTORY + dataset_dir):
        file_path = RAW_SIGNAL_DIRECTORY + dataset_dir + '/' + file
        if file == 'config.txt':
            f = open(file_path)
            config = []
            for line in f:
               config.append(line.replace('\n',''))

            val = config[0]
            row = int(config[1])
            if len(config)>2:
                val2 = config[2]
        elif (file[-3:] == 'txt') or (file[-3:] == 'mat') or (file == 'train.csv') or (file == 'test.csv') :
            print('listing file: ' + file)
            full_paths.append(file_path)

    signals = acquire_and_process_signals(full_paths, signal_dim, peak_into_data=peak_into_data,
                                          decimate=decimate, val=val, val2=val2, row=row, smooth_window=smooth_window, regression=regression)

    if len(signals) > 50:
        signals = [signals]

    y_train = np.zeros(len(signals)).tolist()

    for i in range(len(signals)):
        y_train[i] = signals[i][1:] + [0]

    return signals, y_train

import pandas
def get_biometric_signals(signal_dim, dataset_dir, index, peak_into_data=False, decimate=None, smooth_window=10, regression=False):
    [id, id_test, id_train] = get_biometric_question(dataset_dir, index)
    print("Loading biometric data - id: {0} - train: {1} - test {2}".format(id, id_train, id_test))

    data_frame = pandas.read_csv(RAW_SIGNAL_DIRECTORY + dataset_dir + '/train.csv', delimiter=',')
    data = data_frame[data_frame['Device']==id_train][['T', 'X', 'Y', 'Z']].as_matrix()
    [X_train, Y_train] = getacc(data, signal_dim, smooth_window, peak_into_data, decimate)

    data_frame = pandas.read_csv(RAW_SIGNAL_DIRECTORY + dataset_dir + '/test.csv', delimiter=',')
    data = data_frame[data_frame['SequenceId'] == id_test][['T', 'X', 'Y', 'Z']].as_matrix()
    [X_test, Y_test] = getacc(data, signal_dim, smooth_window, peak_into_data, decimate)

    return X_train, Y_train, X_test, Y_test


def get_biometric_question(dataset_dir, index):
    questions = np.genfromtxt(RAW_SIGNAL_DIRECTORY + dataset_dir + '/questions.csv', delimiter=',', skip_header=1)
    [id, id_test, id_train] = questions[index, :]
    return id, id_test, id_train


def getacc(signal, signal_dim, smooth_window, peak_into_data, decimate):
    time_stamps = signal[:, 0] # -> only if necessary
    times = np.diff(np.diff(time_stamps))
    diff_ = np.sum(np.diff(times))
    print("Sum of the second derivative of time for device: {0}".format(diff_))
    x_signal_aux = process_signal(signal[:,1], signal_dim, smooth_window, peak_into_data, decimate, regression=False)
    y_signal_aux = process_signal(signal[:,2], signal_dim, smooth_window, peak_into_data, decimate, regression=False)
    z_signal_aux = process_signal(signal[:,3], signal_dim, smooth_window, peak_into_data, decimate, regression=False)

    X = [x_signal_aux, y_signal_aux, z_signal_aux]
    Y = [np.append(x_signal_aux[1:], [0]), np.append(y_signal_aux[1:], [0]), np.append(z_signal_aux[1:], [0])]

    return X, Y

"""#####################################################################################################################
########################################################################################################################
#########################                           PLOTS                         ######################################
########################################################################################################################
#####################################################################################################################"""


def create_grid_for_graph(signal, step=0.1):
    X, Y = np.meshgrid(np.arange(np.shape(signal)[0]), np.arange(np.shape(signal)[1]))
    X_grid, Y_grid = np.meshgrid(np.arange(0., np.shape(signal)[0], step),
                                 np.arange(0., np.shape(signal)[1], step))
    X, Y = X.transpose(), Y.transpose()
    return griddata(np.array((X.reshape(-1), Y.reshape(-1))).T, signal.reshape(-1).T, (X_grid, Y_grid),
                    method='linear').T, X_grid, Y_grid


def make_cmap(colors, position=None, bit=False, max_colors=256):
    '''
   make_cmap takes a list of tuples which contain RGB values. The RGB
   values may either be in 8-bit [0 to 255] (in which bit must be set to
   True when called) or arithmetic [0 to 1] (default). make_cmap returns
   a cmap with equally spaced colors.
   Arrange your tuples so that the first color is the lowest value for the
   colorbar and the last is the highest.
   position contains values from 0 to 1 to dictate the location of each color.
   '''

    bit_rgb = np.linspace(0, 1, 256)

    if position is None:
        position = np.linspace(0, 1, len(colors))
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = col.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


def get_color(file_number=3):
    # path = "../DataProcessor/printing/ColorMapLib" + str(file_number) + ".txt"
    path = "../utils/data/biosig_colormap/ColorMapLib" + str(file_number) + ".txt"

    data = open(path)
    lines = data.readlines()
    temp = lines[0].split(";")[0].split("[")[1].split(",")
    colors = [(int(temp[0]), int(temp[1]), int(temp[2]))]
    for i in range(1, len(lines) - 1):
        temp = lines[i].split(";")[0].split(",")
        colors.append((int(temp[0]), int(temp[1]), int(temp[2])))
    temp = lines[len(lines) - 1].split("]")[0].split(",")
    colors.append((int(temp[0]), int(temp[1]), int(temp[2])))
    return colors


def plot_gru_simple(model, original_data, predicted_signal, signal_probabilities=None):
    original_data = np.asarray(original_data)
    predicted_signal = np.asarray(predicted_signal)
    y_lim = [np.min([np.min(original_data), np.min(predicted_signal)]) - 1,
             np.max([np.max(original_data), np.max(predicted_signal)]) + 1]

    # Prepare Figure and plots
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[198, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_ylim(y_lim)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_ylim(y_lim)
    # ax3 = fig.add_subplot(313, projection='3d')

    ax1.set_title("Original signal example")
    ax1.set_ylabel('k')
    ax1.set_xlabel('n')
    ax2.set_title("Dreamed signal - training time: {0} milisec".format(model.train_time))
    ax2.set_ylabel('k')
    ax2.set_xlabel('n')

    cax2 = fig.add_subplot(gs[1, 1])

    # PLOT AXIS 1
    ax1.plot(original_data, color="#990000")

    # PLOT AXIS 2
    bounds = np.arange(0, 1., 0.05).tolist()
    ax2.plot(predicted_signal, color="#009900", alpha=0.3)  # predicted signal
    if signal_probabilities is not None:
        Z_grid, X_grid, Y_grid = create_grid_for_graph(signal_probabilities, 0.1)
        norm = mpl.colors.Normalize(vmin=np.min(Z_grid[~np.isnan(Z_grid)]), vmax=np.max(Z_grid[~np.isnan(Z_grid)]))
        im1 = ax2.imshow(Z_grid, interpolation='spline16',
                         extent=[0, len(predicted_signal), 0, model.signal_dim],
                         cmap=mpl.cm.BuPu, norm=norm, aspect='auto', origin='lower', alpha=0.5)
        ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        fig.colorbar(im1, cax=cax2)

    plt.subplots_adjust(0.04, 0.04, 0.94, 0.96, 0.08, 0.14)
    plt.show()


def plot_gru_only_predicted(model, predicted_signal, show=True):
    predicted_signal = np.asarray(predicted_signal)
    y_lim = [np.min(predicted_signal) - 1,
             np.max(predicted_signal) + 1]

    # Prepare Figure and plots
    fig = plt.figure()

    plt.title("Dreamed signal")
    plt.ylabel('k')
    plt.xlabel('n')

    # PLOT AXIS 1
    plt.plot(predicted_signal)
    if show:
        plt.show()


def plot_gru_data(model, original_data, predicted_signal, signal_probabilities, hidden_layer_data, step):
    original_data = np.asarray(original_data)
    predicted_signal = np.asarray(predicted_signal)
    y_lim = [np.min([np.min(original_data), np.min(predicted_signal)]) - 1,
             np.max([np.max(original_data), np.max(predicted_signal)]) + 1]
    # y_lim = [0,
    #          64]

    # Prepare Figure and plots
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2, width_ratios=[198, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_ylim(y_lim)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_ylim(y_lim)
    # ax3 = fig.add_subplot(313, projection='3d')
    ax3 = fig.add_subplot(gs[2, 0])
    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])

    ax1.set_title("Original signal example")
    ax1.set_ylabel('k')
    ax1.set_xlabel('n')
    ax2.set_title("Dreamed signal example and probabilities")
    ax2.set_ylabel('k')
    ax2.set_xlabel('n')
    ax3.set_title("GRU hidden layer outputs")
    ax3.set_ylabel('Hidden layer')
    ax3.set_xlabel('n')

    # PLOT AXIS 1
    ax1.plot(original_data, color="#990000")

    # PLOT AXIS 2
    bounds = np.arange(0, 1., 0.05).tolist()
    ax2.plot(predicted_signal, color="#009900", alpha=0.3)  # predicted signal
    Z_grid, X_grid, Y_grid = create_grid_for_graph(signal_probabilities, step)
    norm = mpl.colors.Normalize(vmin=np.min(Z_grid[~np.isnan(Z_grid)]), vmax=np.max(Z_grid[~np.isnan(Z_grid)]))
    im1 = ax2.imshow(Z_grid, interpolation='spline16',
                     extent=[0, len(predicted_signal), 0, model.signal_dim],
                     cmap=mpl.cm.BuPu, norm=norm, aspect='auto', origin='lower', alpha=0.5)
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im1, cax=cax2)

    # PLOT AXIS 3


    X_grid, Y_grid = np.meshgrid(np.arange(np.shape(hidden_layer_data)[0]), np.arange(np.shape(hidden_layer_data)[1]))
    Z_grid = hidden_layer_data.T
    Z_grid, X_grid, Y_grid = create_grid_for_graph(hidden_layer_data, step)

    norm = mpl.colors.Normalize(vmin=np.min(Z_grid[~np.isnan(Z_grid)]), vmax=np.max(Z_grid[~np.isnan(Z_grid)]))
    im2 = ax3.imshow(Z_grid, interpolation='none',
                     extent=[0, len(predicted_signal), 0, model.hidden_dim],
                     cmap=mpl.cm.rainbow, norm=norm, aspect='auto', origin='lower', alpha=1)
    ax3.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax3.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im2, cax=cax3)

    plt.subplots_adjust(0.04, 0.04, 0.94, 0.96, 0.08, 0.14)
    plt.show()


def plot_gru_gates_data(model, predicted_signal, reset_layer_data, update_layer_data, step):
    predicted_signal = np.asarray(predicted_signal)
    y_lim = [np.min(np.min(predicted_signal)) - 1,
             np.max(np.max(predicted_signal)) + 1]
    y_lim = [0,
             64]

    cmp = make_cmap(get_color(), bit=True)
    # Prepare Figure and plots
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2, width_ratios=[198, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_ylim(y_lim)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_ylim(y_lim)
    # ax3 = fig.add_subplot(313, projection='3d')
    ax3 = fig.add_subplot(gs[2, 0])
    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])

    ax1.set_title("Dreamed signal example")
    ax1.set_ylabel('k')
    ax1.set_xlabel('n')
    ax2.set_title("GRU 3 Reset Layers")
    ax2.set_ylabel('Hidden layer')
    ax2.set_xlabel('n')
    ax3.set_title("GRU 3 Update Layers")
    ax3.set_ylabel('Hidden layer')
    ax3.set_xlabel('n')

    # PLOT AXIS 1
    ax1.plot(predicted_signal, color="#009900", alpha=0.3)  # predicted signal

    # PLOT AXIS 2
    bounds = np.arange(0, 1., 0.05).tolist()

    X_grid, Y_grid = np.meshgrid(np.arange(np.shape(reset_layer_data)[0]), np.arange(np.shape(reset_layer_data)[1]))
    Z_grid = reset_layer_data.T
    Z_grid, X_grid, Y_grid = create_grid_for_graph(reset_layer_data, step)

    norm = mpl.colors.Normalize(vmin=np.min(Z_grid[~np.isnan(Z_grid)]), vmax=np.max(Z_grid[~np.isnan(Z_grid)]))
    im2 = ax3.imshow(Z_grid, interpolation='none',
                     extent=[0, len(predicted_signal), 0, model.hidden_dim],
                     cmap=cmp, norm=norm, aspect='auto', origin='lower', alpha=1)
    ax3.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax3.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im2, cax=cax3)

    # PLOT AXIS 3


    X_grid, Y_grid = np.meshgrid(np.arange(np.shape(update_layer_data)[0]), np.arange(np.shape(update_layer_data)[1]))
    Z_grid = update_layer_data.T
    Z_grid, X_grid, Y_grid = create_grid_for_graph(update_layer_data, step)

    norm = mpl.colors.Normalize(vmin=np.min(Z_grid[~np.isnan(Z_grid)]), vmax=np.max(Z_grid[~np.isnan(Z_grid)]))
    im2 = ax3.imshow(Z_grid, interpolation='none',
                     extent=[0, len(predicted_signal), 0, model.hidden_dim],
                     cmap=cmp, norm=norm, aspect='auto', origin='lower', alpha=1)
    ax3.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax3.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im2, cax=cax3)

    plt.subplots_adjust(0.04, 0.04, 0.94, 0.96, 0.08, 0.14)
    plt.show()


def plot_gru_gates_data(model, predicted_signal, hidden_layer_data, reset_layer_data, update_layer_data, step):
    predicted_signal = np.asarray(predicted_signal)
    y_lim = [np.min(np.min(predicted_signal)) - 1,
             np.max(np.max(predicted_signal)) + 1]
    # y_lim = [0,
    #          64]

    cmp = make_cmap(get_color(), bit=True)
    # Prepare Figure and plots
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 2, width_ratios=[198, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_ylim(y_lim)
    ax2 = fig.add_subplot(gs[1, 0])

    # ax3 = fig.add_subplot(313, projection='3d')
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])

    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])
    cax4 = fig.add_subplot(gs[3, 1])

    ax1.set_title("Dreamed signal example")
    ax2.set_title("GRU 3 Output Layers")
    ax3.set_title("GRU 3 Reset Layers")
    ax4.set_title("GRU 3 Update Layers")

    # PLOT AXIS 1
    ax1.plot(predicted_signal, color="#009900", alpha=0.3)  # predicted signal
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    # PLOT AXIS 2
    X_grid, Y_grid = np.meshgrid(np.arange(np.shape(hidden_layer_data)[0]), np.arange(np.shape(hidden_layer_data)[1]))
    Z_grid = hidden_layer_data.T
    Z_grid, X_grid, Y_grid = create_grid_for_graph(hidden_layer_data, step)

    norm = mpl.colors.Normalize(vmin=np.min(Z_grid[~np.isnan(Z_grid)]), vmax=np.max(Z_grid[~np.isnan(Z_grid)]))
    im2 = ax2.imshow(Z_grid, interpolation='none',
                     extent=[0, np.shape(hidden_layer_data)[1], 0, np.shape(hidden_layer_data)[0]],
                     cmap=mpl.cm.rainbow, norm=norm, aspect='auto', origin='lower', alpha=1)
    # ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im2, cax=cax2)

    # PLOT AXIS 3

    bounds = np.arange(0, 1., 0.05).tolist()

    X_grid, Y_grid = np.meshgrid(np.arange(np.shape(reset_layer_data)[0]), np.arange(np.shape(reset_layer_data)[1]))
    # reset_layer_data[np.where(reset_layer_data > 0.6)] = 1
    # reset_layer_data[np.where(reset_layer_data < 0.4)] = 0

    Z_grid = reset_layer_data.T
    Z_grid, X_grid, Y_grid = create_grid_for_graph(reset_layer_data, step)
    cmp = mpl.cm.YlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    im3 = ax3.imshow(Z_grid, interpolation='none',
                     extent=[0, np.shape(reset_layer_data)[1], 0, np.shape(reset_layer_data)[0]],
                     cmap=cmp, norm=norm, aspect='auto', origin='lower', alpha=1)
    # ax3.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax3.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im3, cax=cax3)

    # PLOT AXIS 4


    X_grid, Y_grid = np.meshgrid(np.arange(np.shape(update_layer_data)[0]), np.arange(np.shape(update_layer_data)[1]))
    # reset_layer_data[np.where(reset_layer_data > 0.6)] = 1
    # reset_layer_data[np.where(reset_layer_data < 0.4)] = 0
    Z_grid = update_layer_data.T
    Z_grid, X_grid, Y_grid = create_grid_for_graph(update_layer_data, step)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    im4 = ax4.imshow(Z_grid, interpolation='none',
                     extent=[0, np.shape(update_layer_data)[1], 0, np.shape(update_layer_data)[0]],
                     cmap=cmp, norm=norm, aspect='auto', origin='lower', alpha=1)
    # ax4.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax4.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    fig.colorbar(im4, cax=cax4)

    plt.subplots_adjust(0.04, 0.04, 0.94, 0.96, 0.08, 0.14)
    plt.show()


def plot_gru_gates_in_graph(model, predicted_signal, hidden_layer_data, reset_layer_data, update_layer_data, step):
    predicted_signal = np.asarray(predicted_signal)
    y_lim = [np.min(np.min(predicted_signal)) - 1,
             np.max(np.max(predicted_signal)) + 1]
    # y_lim = [0,
    #          64]

    fig = plt.figure()
    cmap = mpl.cm.rainbow
    # gs = gridspec.GridSpec(np.shape(reset_layer_data)[0], 4, width_ratios=[66, 66, 66, 2])
    # for i in range(0, np.shape(reset_layer_data)[0]):
    #     ax1 = fig.add_subplot(gs[i, 0])
    #     ax2 = fig.add_subplot(gs[i, 1])
    #     ax3 = fig.add_subplot(gs[i, 2])
    #     ax4 = fig.add_subplot(gs[i, 3])
    #
    #     for j in range(len(predicted_signal)):
    #         ax1.plot(j, predicted_signal[j], '.', color=cmap(reset_layer_data[i, j]+update_layer_data[i, j]))
    #         ax2.plot(j, predicted_signal[j], '.', color=cmap(0.5+update_layer_data[i, j]/2))
    #         ax3.plot(j, predicted_signal[j], '.', color=cmap(hidden_layer_data[i, j]+1))
    #
    #
    #         cbar = mpl.colorbar.ColorbarBase(ax4, cmap=cmap,
    #                                          norm=mpl.colors.Normalize(vmin=-1, vmax=1))
    #         cbar.set_clim(-1, 1)

    gs = gridspec.GridSpec(np.shape(reset_layer_data)[0], 3, width_ratios=[int(198 / 2), int(198 / 2), 2])
    for i in range(0, np.shape(reset_layer_data)[0]):
        ax1 = fig.add_subplot(gs[i, 0])
        # ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 1])
        ax4 = fig.add_subplot(gs[i, 2])

        for j in range(len(predicted_signal)):
            ax1.plot(j, predicted_signal[j], '.',
                     color=cmap((reset_layer_data[i, j] + update_layer_data[i, j] * 2) / 3))
            # ax2.plot(j, predicted_signal[j], '.', color=cmap(0.5 + update_layer_data[i, j] / 2))
            ax3.plot(j, predicted_signal[j], '.', color=cmap((hidden_layer_data[i, j] + 1) / 2))

            cbar = mpl.colorbar.ColorbarBase(ax4, cmap=cmap,
                                             norm=mpl.colors.Normalize(vmin=-1, vmax=1))
            cbar.set_clim(-1, 1)

    plt.subplots_adjust(0.04, 0.04, 0.94, 0.96, 0.08, 0.14)
    plt.show()


def plot_past_gru_data(predicted_signals, predicted_names):
    # Prepare Figure and plots
    fig = plt.figure()

    N = len(predicted_signals)
    M = len(predicted_signals[0])

    axis = list(range(len(predicted_signals)))
    cmp = make_cmap(get_color(), bit=True, max_colors=N)

    if M < 4:
        gs = gridspec.GridSpec(M, N)
        for i in range(N):
            for j in range(M):
                axis = fig.add_subplot(gs[j, i])
                axis.plot(predicted_signals[i][j], color=cmp(i / N))
                # axis.set_ylim(y_lim)
                axis.set_ylabel('k')
                axis.set_xlabel('n')
                axis.set_title(predicted_names[i][j])

    else:
        gs = gridspec.GridSpec(N, 1)
        for i in range(N):
            axis = fig.add_subplot(gs[i])
            axis.plot(predicted_signals[i], color=cmp(i / N))
            # axis.set_ylim(y_lim)
            axis.set_ylabel('k')
            axis.set_xlabel('n')
            axis.set_title(predicted_names[i])

    # fig.tight_layout()
    plt.show()


"""#####################################################################################################################
########################################################################################################################
#########################           DATASET FOR CROSS VALIDATION                       #################################
########################################################################################################################
#####################################################################################################################"""


def make_training_sets(batches, percentage_of_train):
    train_indexes, test_indexes = train_test_split(list(range(np.shape(batches)[0])), test_size=1 - percentage_of_train)

    train = batches[train_indexes, :]
    test = batches[test_indexes, :]

    return train_indexes, test_indexes, train, test

def plot_confusion_matrix(confusion_matrix, labels_pred, labels_true, title='Confusion matrix' , cmap=plt.cm.Reds,
                          cmap_text=plt.cm.Reds_r, no_numbers=False, norm=False, N_Windows=None):
    # plt.tight_layout()

    N = np.shape(confusion_matrix)[0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    TOTAL = 0
    for line, i in zip(confusion_matrix, range(N)):
        TP += line[i]
        FP += np.sum(confusion_matrix[np.arange(N) != i, i])
        TN += np.sum(confusion_matrix[np.arange(N) != i, np.arange(N) != i])
        FN += np.sum(line) - line[i]
        TOTAL += np.sum(confusion_matrix)

    ACC = 100 * (TP + TN) / TOTAL
    # acc = 100 * np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
    sens = 100 * TN / (TN + FN)
    spec = 100 * TN / (TN + FP)

    fig, ax = plt.subplots()
    ax = prepare_confusion_matrix_plot(ax, confusion_matrix, labels_pred, labels_true, cmap, cmap_text, no_numbers,
                                       norm, N_Windows)

    ax.annotate('Accurancy of {0:.1f}%'.format(ACC),
                xy=(0.5, 0), xytext=(0, 60),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=15, ha='center', va='bottom')

    ax.annotate('Specificity of {0:.1f}%'.format(spec),
                xy=(0.5, 0), xytext=(0, 35),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=15, ha='center', va='bottom')

    ax.annotate('Sensitivity of {0:.1f}%'.format(sens),
                xy=(0.5, 0), xytext=(0, 10),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=15, ha='center', va='bottom')

    # ax = prepare_confusion_pie(ax, confusion_matrix)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def plot_confusion_matrix_with_pie(confusion_matrix, labels_pred, labels_true, rejection=None, title='Confusion matrix',
                                   cmap=plt.cm.Reds, cmap_text=plt.cm.Reds_r, no_numbers=False,norm=False,
                                   N_Windows=None):

    fig, ax = plt.subplots(1, 2)
    ax1 = prepare_confusion_matrix_plot(ax[1], confusion_matrix, labels_pred, labels_true, cmap, cmap_text, no_numbers, norm, N_Windows)
    ax2 = prepare_confusion_pie(ax[0], confusion_matrix, cmap, rejection)
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    kwargs = dict(y=0.93,size=36, fontweight='heavy', va='center', color=cmap(0.9), family="lato")
    fig.suptitle(title, **kwargs)

    plt.show()

def prepare_confusion_matrix_plot(ax, confusion_matrix, labels_pred, labels_true, cmap,
                                  cmap_text, no_numbers, norm, N_Windows):

    if norm:
        for i in range(np.shape(confusion_matrix)[0]):
            confusion_matrix[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])

    # plt.tight_layout()
    # if len(labels_pred) == 2:
    #     labels_pred = ["", labels_pred[0], "", labels_pred[1]]
    # else:
    #     labels_pred = np.array([["", label] for label in labels_pred])
    #     labels_pred = labels_pred.flatten()
    # if len(labels_true) == 2:
    #     labels_true = ["", labels_true[0], "", labels_true[1]]
    # else:
    #     labels_true = np.array([["", label] for label in labels_true])
    #     labels_true = labels_true.flatten()

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
                value = int(confusion_matrix[i, j])
                value_ = str(int(confusion_matrix[i, j]))
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


def prepare_confusion_pie(ax, confusion_matrix, cmap = ["#3366CC", "#79BEDB", "#E84150", "#FFB36D"], rejection = None):
    # plot pi:e chart---------------------------------------------------------------------------------------------
    labels = 'TP', 'TN', 'FP', 'FN'
    font0 = getFontProperties('medium', 'monospace', 25, style = 'italic')

    TP = confusion_matrix[0,0] / np.sum(confusion_matrix)
    TN = confusion_matrix[1,1] / np.sum(confusion_matrix)
    FP = confusion_matrix[1,0] / np.sum(confusion_matrix)
    FN = confusion_matrix[0,1] / np.sum(confusion_matrix)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accurancy = (TP + TN)/(TP + FP + FN + TN)

    perc = [TP, TN, FP, FN]
    explode = (0, 0, 0.1, 0.1)  # explode a slice if required
    patches, texts, autotexts = ax.pie(perc, colors=[cmap(TP/max(perc)),cmap(TN/max(perc)),cmap(FP/max(perc)),cmap(FN/max(perc))], shadow=False, explode=explode, autopct='%1.1f%%',
                                        startangle=180)  # draw a circle at the center of pie to make it look like a donut

    for p, at, pc in zip(patches, autotexts, perc):
        p.set_linewidth(3)
        p.set_alpha(0.8)
        p.set_edgecolor('lightgray')
        at.set_fontproperties(font0)
        at.set_size(25 * np.log10(100 * pc + 2))
        at.set_path_effects([pte.Stroke(linewidth=0.5, foreground='slategray'), pte.Normal()])
        at.set_color('black')

        ax.legend(patches, labels, loc='best', fontsize=15)
        centre_circle = plt.Circle((0, 0), 0.35, color='lightgray', fc='white', linewidth=3)
        ax.add_artist(centre_circle)
        text = 'Sensitivity = ' + str(round(sensitivity * 100.0, 2)) + ' %' + '\n' + 'Specificity = ' + str(
            round(specificity * 100.0, 2)) + ' %' + '\n' + 'Accuracy = ' + str(round(accurancy * 100.0, 2)) + ' %'
        kwargs = dict(size=15, fontweight='medium', va='center', color='slategray')
        txt = ax.text(0, 0, text, ha='center', **kwargs)
        txt.set_path_effects([pte.Stroke(linewidth=0.5, foreground='white'), pte.Normal()])

        ax.axis('equal')
        # gs.tight_layout(fig)

    if rejection is not None:
        rejection = round(FN * 100.0 , 2)
        txt = ax.text(0.5, 0.9, str(rejection)+"%", ha='center', **kwargs)
        txt.set_path_effects([pte.Stroke(linewidth=2, foreground='white'), pte.Normal()])



def getFontProperties(weight, family, size, style=None):
    # Specify Font Properties of text-------------------------------------------------------------------
    font = FontProperties()
    font.set_weight(weight)
    font.set_family(family)
    font.set_size(size)

    if style is not None:
        font.set_style(style)

    return font


def segment_signal(signal, W, overlap=0.5, N_Windows=None, start_index=0):
    """
    This method returns a matrix with window segments with a overlap fraction
    :param signal:
            input signal vector to segment
    :param W:
            size of desired window
    :param overposition:
            window slide overlap fraction (between 0 and 1):
                overlap = 0: makes the window slide one sample;
                overlap = 1: no overlap
    :return:
        matrix with all windows of the segmented signal ([number of windows] X [window_size])
    """

    step = int(W*overlap)
    if overlap == 0:
        step = 1

    end_indexes = list(range(start_index+W, len(signal), step))
    if N_Windows is None:
        N_Windows = len(end_indexes)
    else:
        limit = start_index + (step * N_Windows + W)
        end_indexes = list(range(start_index + W, limit, step))

    limit = step * N_Windows + W
    # print(limit)
    if limit > len(signal):
        end_indexes = list(range(W, len(signal), step))
        N_Windows = len(end_indexes)

    segment_matrix = np.zeros((N_Windows, W))
    y = np.zeros_like(segment_matrix)
    for index in range(N_Windows):
        # print(index)
        segment_matrix[index, :] = signal[end_indexes[index]-W:end_indexes[index]]
        y[index, :] = signal[end_indexes[index]-W+1: end_indexes[index]+1]
    # print(end_indexes[-1])

    return segment_matrix, y, N_Windows, end_indexes[-1]


def segment_matrix(signal, W, overlap=0.5, N_Windows=None, start_index=0):
    """
    This method returns a tensor with window segments with a overlap fraction
    :param signal:
            input signal matrix to segment, each channel should be each line
    :param W:
            size of desired window
    :param overposition:
            window slide overlap fraction (between 0 and 1):
                overlap = 0: makes the window slide one sample;
                overlap = 1: no overlap
    :return:
        matrix with all windows of the segmented signal ([number of windows] X [window_size])
    """
    step = int(W*overlap)
    if overlap == 0:
        step = 1

    N_dimension = len(np.shape(signal))
    N = np.shape(signal)[N_dimension-1] - 1  # Number of samples in the end of processing
    if N_dimension == 1:
        signal = np.reshape(signal, (N_dimension, N+1))

    end_indexes = list(range(start_index+W, N, step))
    if N_Windows is None:
        N_Windows = len(end_indexes)
    else:
        limit = start_index + (step * N_Windows + W)
        end_indexes = list(range(start_index + W, limit, step))

    limit = step * N_Windows + W
    # print(limit)
    if limit > N:
        end_indexes = list(range(W, N, step))
        N_Windows = len(end_indexes)

    X_segment_matrix, Y_segment_matrix = np.zeros((N_dimension, N_Windows, W)), np.zeros((N_dimension, N_Windows, W))
    X = signal[:, :-1]
    Y = signal[:, 1:]
    for index in range(N_Windows):
        X_segment_matrix[:, index, :] = X[:, end_indexes[index] - W:end_indexes[index]]
        Y_segment_matrix[:, index, :] = Y[:, end_indexes[index] - W:end_indexes[index]]
    # print(end_indexes[-1])

    return X_segment_matrix, Y_segment_matrix


def get_models(signals_models, index=None):
    models = []

    if index is not None:
        signals_models = [signals_models[index]]

    for model_info in signals_models:
        model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name)
        model.load(signal_name=model_info.name, filetag=model.get_file_tag(model_info.DS,
                                                                              model_info.t),
                   dir_name=model_info.directory)
        models.append(model)

    return models


def get_signals_tests(signals_tests, Sd=64, index=None, regression=False, type=None, noisy_index=None, peak_into_data=False):
    history =[0]
    s = -1

    if type is not None:
        signals_tests_temp = []
        for signal_info in signals_tests:
            if type == signal_info.type:
                signals_tests_temp.append(signal_info)

            signals_tests = signals_tests_temp
    elif index is not None:
        signals_tests = [signals_tests[index]]


    N_Signals = len(signals_tests)
    signals = [list(range(N_Signals)),list(range(N_Signals))]

    for signal_info in signals_tests:

        s += 1
        i = 0
        print("Loading " + signal_info.name, end=': ')
        if signal_info.type == "ecg":
            if history[-1] == "ecg":
                print("Processing ecg " + str(signal_info.index-1), end=': ')
                i = signal_info.index-1
            else:
                X_train, Y_train = get_fantasia_dataset(Sd, list(range(1, 21)), signal_info.directory,
                                                    peak_into_data=False, regression=regression)
        elif signal_info.type == "ecg noise":
            X_train, Y_train = get_fantasia_noisy_data(Sd, list(range(1, 21)), noisy_index, signal_info.directory,
                                                        peak_into_data=peak_into_data, regression=regression)

            signals[0] = np.asarray(X_train)
            signals[1] = np.asarray(Y_train)
            return signals
        elif signal_info.type == "resp":
            if history[-1] == "resp":
                index = signal_info.index - 1
                print("Processing resp data " + str(index), end=': ')
            else:
                X_train, Y_train = get_fantasia_dataset(Sd, list(range(1, 21)), signal_info.directory,
                                                    peak_into_data=False, regression=regression)
        elif signal_info.type == "day hrv":
            val = signal_info.file_name[:-5]+'s'
            signals = get_day_dataset(Sd, signal_info.directory, file_name=signal_info.file_name,
                                               peak_into_data=peak_into_data, regression=regression, val=val)
            if index is not None:
                return signals[:][index]
            else:
                return signals
        elif signal_info.type == "day":
            signals = get_day_dataset(Sd, signal_info.directory, file_name=signal_info.file_name,
                                               peak_into_data=peak_into_data, regression=regression)
            if index is not None:
                return signals[:][index]
            else:
                return signals
        elif signal_info.type == "emg":
            X_train, Y_train = get_signals(Sd, signal_info.directory,
                                           peak_into_data=False, decimate=2, val='', row=6, regression=regression)
        elif signal_info.type == "eeg":
            if history[-1] == "eeg":
                i = signal_info.index
            else:
                X_train, Y_train = get_signals(Sd, signal_info.directory,
                                           peak_into_data=False, regression=regression)
        elif signal_info.type == "biometric":
            index = signal_info.index
            return get_biometric_signals(Sd, signal_info.directory, index,
                                               peak_into_data=False, regression=regression)

        elif signal_info.type == "gsr":
            if history[-1] == "gsr":
                i += 1
            else:
                i = 0
                X_train, Y_train = get_signals(Sd, signal_info.directory,
                                           peak_into_data=False, regression=regression)

        history.append(signal_info.type)

        N = len(X_train)
        # if X_train.__class__ is list:
        #     N = len(X_train[i])

        signals[0][s] = np.asarray(X_train[i])
        signals[1][s] = np.asarray(Y_train[i])

    return signals


def get_random_batch(X, Y, window_size, batch_size, overlap=0.25, start_index=0):
    x_windows, y_end_values, N_batches, last_index = segment_signal(X, window_size, overlap=overlap,
                                                                    start_index=start_index)
    y_windows, y_end_values, N_batches, last_index = segment_signal(Y, window_size, overlap=overlap,
                                                                    start_index=start_index)

    return randomize_batch(x_windows, y_windows, batch_size)


def randomize_batch(x_windows, y_windows, batch_size=None):
    if len(np.shape(x_windows))>2:
        N_batches = np.shape(x_windows)[1]
    else:
        N_batches = np.shape(x_windows)[0]

    window_indexes = np.random.permutation(N_batches)  # randomly select windows

    if batch_size is None:
        batch_size = N_batches

    if len(np.shape(x_windows)) <= 2:
        return x_windows[window_indexes[0:batch_size], :], y_windows[window_indexes[0:batch_size], :]
    else:
        return x_windows[:, window_indexes[0:batch_size], :], y_windows[:, window_indexes[0:batch_size], :]