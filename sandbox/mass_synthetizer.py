
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysMBGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

# get_fantasia_dataset('Fantasia/ECG/mat/',64)
# get_fantasia_dataset(64, [1], 'Fantasia/ECG/mat/')

def synthetize(models, uncertaintly=0.01, filename="synthesized_signals_1024.npz"):
    mini_bs=256
    plt.ion()
    signals = []
    signal2Model = Signal2Model(models[0].dataset_name, models[0].directory, signal_dim=models[0].Sd,
                                hidden_dim=models[0].Hd, mini_batch_size=mini_bs)
    model = GRU.LibphysMBGRU(signal2Model)
    for model_info in models:
        model.model_name = model_info.dataset_name
        model.load(dir_name=model_info.directory)
        print("Processing " + model_info.name)

        # plt.ion()
        # font = {'family': 'lato',
        #         'weight': 'bold',
        #         'size': 40}
        #
        # matplotlib.rc('font', **font)
        x = [0]
        i = 0

        signal = np.random.randint(0, 63, size=(mini_bs, 1), dtype=np.int32)
        window_size = 512
        fz = 250
        N = 512
        # while True:
        for i in range(N):
            # print(i)
            y = model.generate_online_predicted_vector(signal, window_size, uncertaintly=uncertaintly)
            signal.append(y)
        plt.clf()
        plt.plot(signal)
        plt.pause(0.05)
        signals.append(np.array(signal))

        np.savez("img/"+model_info.dataset_name + ".npz", synth_signal=signal, probability=prob)
    np.savez(filename, synth_signals=signals, probabilities=probabilities)

# filename="synthesized_signals_64.npz"
filename = "ecg_3.npz"
models = [db.ecg_64_models[2]]

synthetize(models, uncertaintly=0.01, filename=filename)

synth_signals = np.load(filename)["synth_signal"]
probabilities = np.load(filename)["probability"]
originals = [np.load("../data/FANTASIA_ECG[64].npz")['x_train'][2]]
            #, np.load("../data/Fantasia_RESP_[64].npz")['x_train'][2]]
             #, np.load("../data/FMH_[64].npz")['x_train'][0]]

for synth, original, probs, model in zip([synth_signals], originals, [probabilities], models):
    plot_gru_simple(model, original[10100:11100], synth[:1000], probs[:1000,:].T)