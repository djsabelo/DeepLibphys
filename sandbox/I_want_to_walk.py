import DeepLearningTryAndError.libphys_GRU as GRU
from DeepLearningTryAndError.utils import prepare_emg_data, plot_gru_data, plot_gru_simple
import matplotlib.pyplot as plt
import numpy as np


# from mayavi import mlab

signal_dim = 64
hidden_dim = 128
signal_name = "provadeesforco_emg_32_1300_300"
signal_directory = 'EMGs'
learning_rate_val = 0.01
batch_size = 64
window_size = 512
number_of_epochs = 5000
N = 1000

X_train, Y_train = prepare_emg_data(signal_name + str(signal_dim), signal_dim, filetoextract='PLUX/provadeesforco',
                                    peak_into_data=False, index=11, row=6, sufix='', txt=True, smooth_window=10)

model = GRU.LibPhys_GRU(signal_dim, hidden_dim=hidden_dim, signal_name=signal_name)# "provadeesforco_emg_400")

# model.load(filetag=model.get_file_tag(32, 1300), dir_name=signal_directory)
# model.signal_name = signal_name
# model.save(filetag=model.get_file_tag(-1, -1), dir_name=signal_directory)
#
#
# # One dataset
# model.train(X_train[0], Y_train[0],
#             first_index=32,
#             save_directory=signal_directory,
#             batch_size=batch_size,
#             window_size=window_size,
#             nepoch_val=number_of_epochs,
#             learning_rate_val=learning_rate_val,
#             decay=0.95,
#             track_loss=True)
#
M = 1024

model.load(filetag=model.get_file_tag(448, 400), dir_name=signal_directory)
start_index = np.random.random_integers(0, len(X_train[0])-N)
if start_index < M:
    start_index = M
predicted_signal, probabilities, outputs, z, r = model.generate_predicted_signal(N, X_train[0][start_index-M:start_index].tolist(), M)

plot_gru_simple(model, X_train[0][start_index:start_index+N], predicted_signal[-N:], probabilities)
