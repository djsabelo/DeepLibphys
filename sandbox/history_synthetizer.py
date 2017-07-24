
from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysSGDGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

def create_history(subject):

    model_info = db.ecg_64_models[subject]
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
    window_size = 512
    fz = 250
    N = 3000
    titles = []
    signals = []
    for epoch in [-1, 20, 30, 50, 80, 410]:
        signal = [np.random.randint(0, 63, size=1)[0]]

        if epoch < 0:
            file_tag = model.get_file_tag(epoch, epoch)
        else:
            file_tag = model.get_file_tag(0, epoch)

        model.load(file_tag=file_tag, dir_name=model_info.directory)
        print("Processing epoch " + str(epoch))
        for i in range(N):
            # print(i)
            sample, _ = model.generate_online_predicted_signal(signal, window_size, uncertaintly=0.01)
            signal.append(sample)
            x.append( (i + 1) / fz)

        titles.append("Epoch {0}".format(epoch))
        signals.append(np.array(signal))


    np.savez("../data/history_of_{0}.npz".format(subject), titles=titles, signals=signals, t=x)
    return [titles, signals, x]

# titles, signals, t = create_history(6)
file = np.load("../data/history_of_{0}.npz".format(6))
titles, signals, t = file["titles"], file["signals"], file["t"]
plot_past_gru_data(signals[:, :2000], titles)