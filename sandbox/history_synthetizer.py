
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysSGDGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

def create_history(subject):

    model_info = db.ecg_1024_clean_models[subject]
    signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                    hidden_dim=model_info.Hd)
    model = GRU.LibphysSGDGRU(signal2Model)

    # plt.ion()
    # font = {'family': 'lato',
    #         'weight': 'bold',
    #         'size': 40}
    #
    # matplotlib.rc('font', **font)
    x = [0]
    i = 0
    signal = [10]
    window_size = 512
    fz = 250
    N = 5000
    titles = []
    signals = []
    for epoch in [-1, 5, 10, 15, 50, 100, -5]:
        signal = [np.random.randint(0, 63, size=1)[0]]
        try:
            if epoch < 0:
                file_tag = model.get_file_tag(epoch, epoch)
            else:
                file_tag = model.get_file_tag(0, epoch)

            model.load(file_tag=file_tag, dir_name=model_info.directory)
            print("Processing epoch " + str(epoch))
            for i in range(N):
                # print(i)
                signal.append(model.generate_online_predicted_signal(signal, window_size))
                x.append((i+1)/fz)

            titles.append("Epoch {0}".format(epoch))
            signals.append(np.array(signal))
        except FileNotFoundError:
            pass

    np.savez("../data/history_of_{0}.npz".format(subject), titles=titles, signals=signals, t=x)
    return [titles, signals, x]

titles, signals, t = create_history(6)

plot_past_gru_data(signals, titles)