
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np
import DeepLibphys.models.LibphysSGDGRU as GRU

i = 0
signal = np.random.randint(0, 63, size=1)
model_info = db.emg_64_models[i]
signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                hidden_dim=model_info.Hd)
model = GRU.LibphysSGDGRU(signal2Model)

file_tag = model.get_file_tag(-5, -5)
model.load(dir_name=model_info.directory)
print("Processing " + model_info.name)

# plt.ion()
font = {'family': 'lato',
        'weight': 'bold',
        'size': 40}

matplotlib.rc('font', **font)
x = [0]
i = 0
x_train = np.load("../data/FMH_[64].npz")['x_train'][i]
signal = [np.random.randint(0, 63, size=1)[0]]#x_train[1000:1256].tolist()#[np.random.randint(0, 63, size=1)[0]]
window_size = 512
fz = 250
N = 2000
signal, probs = model.generate_predicted_signal(N, signal, window_size, uncertaintly=0.01)
filename = "../data/EMG_[1.512].npz"
np.savez(filename, synth_signal=signal, probability=probs)
signal, probs = np.load(filename)["synth_signal"], np.load(filename)["probability"]
plt.title("Physiological Signal Synthesizer", fontsize=32)
    # if i >= N:
    #     plt.xlim([(i-N)/fz, i/fz])
# accx = get_fmh_emg_datset(20000, row=8, example_index_array=[i])[0]
# accx = accx[10100:12100]-np.min(accx[10100:12100])
# accx = accx*64/np.max(accx)
# accy = get_fmh_emg_datset(20000, row=9, example_index_array=[i])[0]
# accy = accy[10100:12100]-np.min(accy[10100:12100])
# accy = accy*64/np.max(accy)
#
# accz = get_fmh_emg_datset(20000, row=10, example_index_array=[i])[0]
# accz = accz[10100:12100]-np.min(accz[10100:12100])
# accz =

# plt.plot(accx, 'r', accy, 'g', accz, 'b')
# plt.show()
x_train = x_train[10100:10100+len(signal)]
# plot_gru_simple(model_info, x_train[10100:12100], signal, np.array(probs).T)
plot_gru_simple(model_info, x_train, signal, probs.T)

    # plt.scatter(i/250, signal[-1], marker='.', linestyle='--')
# plt.pause(0.01)
# i += 1