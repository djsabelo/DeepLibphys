import matplotlib.pyplot as plt

import DeepLibphys.models.libphys_GRU as GRU
from DeepLibphys.utils.functions.common import plot_gru_only_predicted, plot_past_gru_data, get_signals_tests, plot_gru_simple
from DeepLibphys.utils.functions import database as db
import seaborn

N = 1024
example_index = [7]
signal_dim = 64
hidden_dim = 256
signal_name = "RGRU_"
signal_directory = 'ECGs_FANTASIA_[256.256]'

# Load Model
index = 8
signals_tests = db.signal_tests
signals_models = db.ecg_models
signals = get_signals_tests(signals_tests, signals_models[0].Sd, index=index, regression=False)

model = GRU.LibPhys_GRU(signal_dim=64, signal_name=signals_models[index].dataset_name, hidden_dim=signals_models[index].Hd)
model.load(filetag=model.get_file_tag(-5, -5), dir_name=signals_models[index].directory)
predicted_signal = model.generate_predicted_signal(N, [40], 256)

plot_gru_simple(model, signals[0], predicted_signal)


# signals_tests = db.signal_tests
# signals_models = db.signal_models
# signals = get_signals_tests(signals_tests, signals_models, index=7, regression=True)
#
# signals_tests = db.signal_tests
# signals_models = db.signal_models
# print(predicted_signal)
# plot_gru_only_predicted(model, predicted_signal)

# signals = get_signals_tests(signals_tests, signals_models)

# for i in range(3,7):
#     signal = Signal2Model('ecg_' + str(i), signal_directory, batch_size=batch_size)
#     model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name='ecg_'+str(i))
#     model.load(filetag=model.get_file_tag(-5, -5), dir_name=signal_directory)
#     predicted_signal = model.generate_predicted_signal(N, [40], 256)
#     plot_gru_only_predicted(model, predicted_signal, False)

# N = 2048
# example_index = 7
# signal_dim = 64
# hidden_dim = 256
# signal_name = "ecg_7_history"
# signal_directory = 'ECGs_FANTASIA_[256.256]'
#
# datasets = [[-1, -1], [0, 10], [0, 20], [0, 50], [0, 100], [0, 390]]
# names = list(range(len(datasets)))
# predicted_signals = list(range(len(datasets)))
# model = GRU.LibPhys_GRU(signal_dim=signal_dim, hidden_dim=hidden_dim, signal_name=signal_name)
# # Make predicted signals
# for i in range(len(datasets)):
#     model.load(filetag=model.get_file_tag(datasets[i][0], datasets[i][1]), dir_name=signal_directory)
#     predicted_signals[i] = model.generate_predicted_signal(N, [40], 256)
# #     names[i] = str(datasets[i]) + " @ " + str(int(model.train_time/3600000)) + " hours"
# #
# plot_past_gru_data(predicted_signals, names)
# # plot_gru_only_predicted(model, predicted_signal)
#
# plt.show()
