
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np
import DeepLibphys.models.LibphysSGDGRU as GRU

i = 2
signal = np.random.randint(0, 63, size=1)
model_info = db.emg_64_models[i]
# signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
#                                 hidden_dim=model_info.Hd)
# model = GRU.LibphysSGDGRU(signal2Model)
#
# file_tag = model.get_file_tag(-5, -5)
# model.load(dir_name=model_info.directory)
print("Processing " + model_info.name)

# plt.ion()
font = {'family': 'lato',
        'weight': 'bold',
        'size': 40}

matplotlib.rc('font', **font)
x = [0]
x_train = np.load("../data/processed/Fantasia_RESP_[64].npz")['x_train'][i]
signal = [np.random.randint(0, 63, size=1)[0]]#x_train[1000:1256].tolist()#[np.random.randint(0, 63, size=1)[0]]
window_size = 512
fz = 250
N = 2000
# signal, probs = model.generate_predicted_signal(N, signal, window_size, uncertaintly=0.01)
filename = "../data/processed/resp_" + str(i+1) + ".npz"
# np.savez(filename, synth_signal=signal, probability=probs)
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

# accz, emgs = get_fmh_emg_datset(20000, row=8, example_index_array=[i])
# accz = get_fmh_emg_datset(20000, row=10, example_index_array=[i])[0]
# accz = accz[10100:12100]-np.min(accz[10100:12100])

# plt.plot(accx, 'r', accy, 'g', accz, 'b')
# plt.show()
signal = signal[500:3500]
probs = probs[500:3500]
x_train = x_train[10000:10000+len(signal)]
# accz = accz[0][10550:10550+len(signal)]
accz = np.load('accz.npz')['accz'][0][10550:10550+len(signal)]
accz = accz - np.min(accz)
accz = accz * 64 / np.max(accz)

probs = probs
# plot_gru_simple(model_info, x_train, signal, np.array(probs).T)
plot_gru_simple_emg(model_info, x_train, [accz, signal], signal, probs.T)

    # plt.scatter(i/250, signal[-1], marker='.', linestyle='--')
# plt.pause(0.01)
# i += 1