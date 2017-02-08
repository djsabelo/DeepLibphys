from BiosignalsDeepLibphys.utils.functions.common import get_fantasia_dataset
from BiosignalsDeepLibphys.utils.functions.libphys_GRU import LibPhys_GRU
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np
SIGNAL_DIRECTORY = '../utils/data/trained/'


N = 1000
signal_dim = 64
hidden_dim = 128
signal_name = "provadeesforco_emg"
signal_directory = 'EMG'
window_size = 2048
fz = 500
signal = [25]

N = 1000
signal_dim = 64
hidden_dim = 64
signal_name = "ecg_10"
signal_directory = 'ECG_10'
window_size = 256
fz = 250
signal = [40]
dataset = 32
epoch = 430


model = LibPhys_GRU(signal_dim, hidden_dim=hidden_dim, signal_name=signal_name)
model.load(dir_name=signal_directory, filetag=model.get_file_tag(dataset, epoch))
plt.ion()
font = {'family': 'lato',
        'weight': 'bold',
        'size': 40}

matplotlib.rc('font', **font)
x = [0]
i = 0
while True:
    # print(i)
    signal.append(model.generate_online_predicted_signal(signal, window_size))
    x.append((i+1)/fz)

    plt.clf()
    plt.ylim([30, 60])
    plt.xlim([0, N / fz])
    plt.title("Physiological Signal Synthesizer", fontsize=32)
    if i >= N:
        plt.xlim([(i-N)/fz, i/fz])
    plt.plot(x, signal)
    # plt.scatter(i/250, signal[-1], marker='.', linestyle='--')
    plt.pause(0.5)
    i += 1




