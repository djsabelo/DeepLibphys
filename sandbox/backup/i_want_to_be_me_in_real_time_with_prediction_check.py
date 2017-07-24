import DeepLibphys.utils.functions.libphys_GRU as GRU
from DeepLibphys.utils.functions.common import get_fantasia_dataset, get_signals, segment_signal, get_models, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys

#CONFUSION_WINDOWS_[W,Z]
W = 256
Z = 10000
filename = "CONFUSION_WINDOWS_["+str(W)+","+str(Z)+"]"


#signals_models = [type (eeg, fantasia, emg or other), directory name, example, name for graph]
signals_tests = [ #["resp", 'Fantasia/RESP/mat/', 19, 900000, "RESP 7"],
                    ["ecg", 'Fantasia/ECG/mat/',  7,  900000, "ECG 7"],
                  ["ecg", 'Fantasia/ECG/mat/', 19,  900000, "ECG 19"],

                ]

npzfile = np.load(filename+".npz")
pred_signals, errors, signals_models, signals_tests = \
    npzfile["pred_signals"], npzfile["errors"], npzfile["signals_models"], npzfile["signals_tests"]

X_train, Y_train = get_fantasia_dataset(signals_models[0]['Sd'], [7], signals_tests[0][1],
                                        peak_into_data=False)
# models = get_models(signals_models)
A = "Synthetised signal"
B = "Error"

print(filename+".npz has been loaded")
fz = 250

plt.figure(A)
plt.title("Physiological Signal Synthesizer", fontsize=32)
x = np.asarray(list(range(Z+1)))/fz


# colors = ['r', 'g', 'b', 'c', 'p']
# plt.plot(x, X_train[0][0:Z+1], label="RAW")
# for m_index in range(len(signals_models)):
#     plt.plot(x, pred_signals[m_index,:], colors[m_index], label=signals_models[m_index]["s_name"])
#
# plt.legend()
# plt.draw()
# plt.figure(B)
#
# for m_index in range(len(signals_models)):
#     plt.plot(x, errors[m_index, :], colors[m_index], label=signals_models[m_index]["s_name"])
#
# plt.legend()
# plt.show()


segments_1 = segment_signal(errors[0,:], 256, 0)
segments_2 = segment_signal(errors[1,:], 256, 0)
segments_3 = segment_signal(errors[2,:], 256, 0)
segments_4 = segment_signal(errors[3,:], 256, 0)
stds_1 = np.std(segments_1[0], axis=1)
stds_2 = np.std(segments_2[0], axis=1)
stds_3 = np.std(segments_3[0], axis=1)
stds_4 = np.std(segments_4[0], axis=1)

stds = np.vstack((stds_1,stds_2,stds_3,stds_4))
loss = np.min(stds)

plot_confusion_matrix

def print_confusion(Mod, Sig, loss_tensor, signals_models, signals_tests):
    labels_model = np.asarray(np.zeros(len(Mod)*2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig)*2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1,len(Mod)*2,2))] = [signals_models[i]["s_name"] for i in Mod]
    labels_signals[list(range(1,len(Sig)*2,2))] = [signals_tests[i][-1] for i in Sig]

    predicted_matrix = np.argmin(loss_tensor[Mod][:, Sig, :], axis = 0)

    sinal_predicted_matrix = np.zeros((len(Sig), len(Mod)))

    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i,:] == j)


    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)
    plot_confusion_matrix(sinal_predicted_matrix, labels_model, labels_signals)# , cmap=cmap)

#
#

