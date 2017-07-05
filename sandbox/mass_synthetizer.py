
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysSGDGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

# get_fantasia_dataset('Fantasia/ECG/mat/',64)
# get_fantasia_dataset(64, [1], 'Fantasia/ECG/mat/')

def synthetize(models, uncertaintly=0.01, filename="synthesized_signals_1024.npz"):
    plt.ion()
    signals = []
    signal2Model = Signal2Model(models[0].dataset_name, models[0].directory, signal_dim=models[0].Sd,
                                hidden_dim=models[0].Hd)
    model = GRU.LibphysSGDGRU(signal2Model)
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

        signal = [np.random.randint(0, 63, size=1)[0]]
        window_size = 512
        fz = 250
        N = 100000
        # while True:
        for i in range(N):
            # print(i)
            signal.append(model.generate_online_predicted_signal(signal, window_size, uncertaintly=uncertaintly))
        plt.clf()
        plt.plot(signal)
        plt.pause(0.05)
        signals.append(np.array(signal))

    np.savez(filename, synth_signals=signals)

filename="synthesized_signals_1024_2.npz"
models = db.ecg_1024_clean_models

synthetize(models, uncertaintly=0.01, filename=filename)

synth_signals = np.load(filename)["synth_signals"]
for signal in synth_signals:
    plt.plot(signal)
    plt.show()