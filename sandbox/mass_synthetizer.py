
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
def sinthetize_signals():
    model_info = db.ecg_clean_models
    signals = []
    for model_info in db.ecg_clean_models:
        signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                        hidden_dim=model_info.Hd)
        model = GRU.LibphysSGDGRU(signal2Model)

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
        window_size = 256
        fz = 250
        N = 90000
        # while True:
        for i in range(N):
            # print(i)
            signal.append(model.generate_online_predicted_signal(signal, window_size))

        signals.append(np.array(signal))

    np.savez("synthesized_signals.npz", synth_signals=signals)


def plot_signals():
    synth_signals = np.load("synthesized_signals.npz")["synth_signals"]

    for sig in synth_signals:
        plt.plot(sig)
        plt.show()

plot_signals()