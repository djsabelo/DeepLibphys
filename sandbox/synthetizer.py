
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysSGDGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

signal = np.random.randint(0, 63, size=1)
model_info = db.ecg_1024_clean_models[6]
signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                hidden_dim=model_info.Hd)
model = GRU.LibphysSGDGRU(signal2Model)

file_tag = model.get_file_tag(-1, -1)
model.load(file_tag=file_tag, dir_name=model_info.directory)
print("Processing " + model_info.name)

plt.ion()
font = {'family': 'lato',
        'weight': 'bold',
        'size': 40}

matplotlib.rc('font', **font)
x = [0]
i = 0
signal = [np.random.randint(0, 63, size=1)[0]]
window_size = 256
fz = 250
N = 1000
while True:
    # print(i)
    signal.append(model.generate_online_predicted_signal(signal, window_size))
    x.append((i+1)/fz)

    plt.clf()
    plt.ylim([0, 64])
    plt.xlim([0, N / fz])
    plt.title("Physiological Signal Synthesizer", fontsize=32)
    if i >= N:
        plt.xlim([(i-N)/fz, i/fz])
    plt.plot(x, signal)
    # plt.scatter(i/250, signal[-1], marker='.', linestyle='--')
    plt.pause(0.01)
    i += 1