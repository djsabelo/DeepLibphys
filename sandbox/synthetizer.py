
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysSGDGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

i = 0
signal = np.random.randint(0, 63, size=1)
model_info = db.emg_64_1024_models[i]
signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                hidden_dim=model_info.Hd)
model = GRU.LibphysSGDGRU(signal2Model)

file_tag = model.get_file_tag(-5, -1)
model.load(dir_name=model_info.directory)
print("Processing " + model_info.name)

# plt.ion()
font = {'family': 'lato',
        'weight': 'bold',
        'size': 40}

matplotlib.rc('font', **font)
x = [0]
i = 0
signal = [np.random.randint(0, 63, size=1)[0]]
window_size = 1024
fz = 250
N = 1000
signal, probs = model.generate_predicted_signal(N, signal, window_size, uncertaintly=0.001)

plt.title("Physiological Signal Synthesizer", fontsize=32)
    # if i >= N:
    #     plt.xlim([(i-N)/fz, i/fz])
x_train = np.load("../data/FMH_[64].npz")['x_train'][i]
plot_gru_simple(model_info, x_train[10100:12100], signal, np.array(probs).T)
    # plt.scatter(i/250, signal[-1], marker='.', linestyle='--')
# plt.pause(0.01)
# i += 1