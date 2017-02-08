import BiosignalsDeepLibphys.utils.functions.libphys_GRU as GRU
from BiosignalsDeepLibphys.utils.functions.common import get_fantasia_dataset, get_signals, segment_signal, get_models, get_signals_tests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys

#CONFUSION_WINDOWS_[W,Z]
W = 256
Z = 3000
filename = "PREDICTION ERROR_["+str(W)+","+str(Z)+"]"

signals_models = [  {"Sd":64, "Hd":256, "name":"ecg_old_fantasia", "dir":"FANTASIA[32x10.256]",
                     "DS":-5,"t":-5,"W":256,"s_name":"ECG_NEW"},
                    {"Sd": 64, "Hd": 256, "name": "resp_[1.2.3.4.5.6.7.8.9.10]_old_fantasia", "dir": "RESP_FANTASIA[1000.256]",
                     "DS":-5, "t": -5, "W": 256, "s_name": "RESP"},
                    {"Sd":64, "Hd":256, "name":"ecg_[1.2.5.6.7.8.9.10]_old_fantasia","dir":"FANTASIA[1000.256]","DS":-5,
                     "t":-5,"W":256,"s_name":"ECG"},
                    {"Sd": 64, "Hd": 256, "name": "eeg_all", "dir": "EEG_Attention[1000.256]","DS":-5, "t":-5, "W":256,
                     "s_name": "EEG"}
                    ]

#signals_models = [type (eeg, fantasia, emg or other), directory name, example, name for graph]
signals_tests  = [  ["emg", 'EMG Cycling',       0,  300000, "EMG BIKE"],
                    ["ecg", 'Fantasia/ECG/mat/', 1,  900000, "ECG 1"],
                    ["ecg", 'Fantasia/ECG/mat/', 2,  900000, "ECG 2"],
                    ["ecg", 'Fantasia/ECG/mat/', 3,  900000, "ECG 3"],
                    ["ecg", 'Fantasia/ECG/mat/', 4,  900000, "ECG 4"],
                    ["ecg", 'Fantasia/ECG/mat/', 5,  900000, "ECG 5"],
                    ["ecg", 'Fantasia/ECG/mat/', 6,  900000, "ECG 6"],
                    ["ecg", 'Fantasia/ECG/mat/', 7,  900000, "ECG 7"],
                    ["ecg", 'Fantasia/ECG/mat/', 8,  900000, "ECG 8"],
                    ["ecg", 'Fantasia/ECG/mat/', 9,  900000, "ECG 9"],
                    ["ecg", 'Fantasia/ECG/mat/', 10, 900000, "ECG 10"],
                    ["ecg", 'Fantasia/ECG/mat/', 11, 900000, "ECG 11"],
                    ["ecg", 'Fantasia/ECG/mat/', 12, 900000, "ECG 12"],
                    ["ecg", 'Fantasia/ECG/mat/', 13, 900000, "ECG 13"],
                    ["ecg", 'Fantasia/ECG/mat/', 14, 900000, "ECG 14"],
                    ["ecg", 'Fantasia/ECG/mat/', 15, 900000, "ECG 15"],
                    ["ecg", 'Fantasia/ECG/mat/', 16, 900000, "ECG 16"],
                    ["ecg", 'Fantasia/ECG/mat/', 17, 900000, "ECG 17"],
                    ["ecg", 'Fantasia/ECG/mat/', 18, 900000, "ECG 18"],
                    ["ecg", 'Fantasia/ECG/mat/', 19, 900000, "ECG 19"],
                    ["other",'DRIVER_GSR',         1,  55800,  "GSR"],
                    ["other",'Fantasia/RESP/mat/', 1,  900000, "RESP 1"],
                    ["resp", 'Fantasia/RESP/mat/', 2,  900000, "RESP 2"],
                    ["resp", 'Fantasia/RESP/mat/', 3,  900000, "RESP 3"],
                    ["resp", 'Fantasia/RESP/mat/', 4,  900000, "RESP 4"],
                    ["resp", 'Fantasia/RESP/mat/', 5,  900000, "RESP 5"],
                    ["resp", 'Fantasia/RESP/mat/', 6,  900000, "RESP 6"],
                    ["resp", 'Fantasia/RESP/mat/', 7,  900000, "RESP 7"],
                    ["resp", 'Fantasia/RESP/mat/', 8,  900000, "RESP 8"],
                    ["resp", 'Fantasia/RESP/mat/', 9,  900000, "RESP 9"],
                    ["resp", 'Fantasia/RESP/mat/', 10, 900000, "RESP 10"],
                    ["resp", 'Fantasia/RESP/mat/', 11, 900000, "RESP 11"],
                    ["resp", 'Fantasia/RESP/mat/', 12, 900000, "RESP 12"],
                    ["resp", 'Fantasia/RESP/mat/', 13, 900000, "RESP 13"],
                    ["resp", 'Fantasia/RESP/mat/', 14, 900000, "RESP 14"],
                    ["resp", 'Fantasia/RESP/mat/', 15, 900000, "RESP 15"],
                    ["resp", 'Fantasia/RESP/mat/', 16, 900000, "RESP 16"],
                    ["resp", 'Fantasia/RESP/mat/', 17, 900000, "RESP 17"],
                    ["resp", 'Fantasia/RESP/mat/', 18, 900000, "RESP 18"],
                    ["resp", 'Fantasia/RESP/mat/', 19, 900000, "RESP 19"],
                    ["eeg",  'EEG_Attention',      0,  429348, "EEG ATT 0"],
                    ["eeg",  'EEG_Attention',      1,  429348, "EEG ATT 1"],
                    ["eeg",  'EEG_Attention',      2,  429348, "EEG ATT 2"],
                    ["eeg",  'EEG_Attention',      3,  429348, "EEG ATT 3"],
                    ["eeg",  'EEG_Attention',      4,  429348, "EEG ATT 4"],
                    ["eeg",   'EEG_Attention',     5,  429348, "EEG ATT 5"],
                    ]



N_Signals = len(signals_tests)
N_Models = len(signals_models)

s = -1;
history = [""]
N_windows = 0
# X_train, Y_train = get_fantasia_dataset(signals_models[0]['Sd'], [7], signals_tests[0][1],
#                                         peak_into_data=False)

models = get_models(signals_models)
signals = get_signals_tests(signals_tests,signals_models)
loss_tensor = np.zeros((N_Models, N_Signals, Z))
m = -1



font = {'family': 'lato',
        'weight': 'bold',
        'size': 40}

matplotlib.rc('font', **font)

i = 0
fz = 250
x = [0]
errors = np.zeros((len(signals_models),N_Signals,Z+1))
pred_signals = np.zeros((len(signals_models),N_Signals,Z+1))

A = "Synthetised signal"
B = "Error"

for s in range(N_Signals):
    indexes = [0, 1]
    signal = signals[s]
    n_samples = Z
    print("Calculating Errors of signal: " + signals_tests[s][4], end="")
    if Z > len(signal):
        n_samples = len(signal)

    for z in range(n_samples):
        for m_index in range(len(models)):
            pred_signals[m_index, s, z+1] = models[m_index].generate_online_predicted_signal(signals[0][indexes[0]:indexes[1]], W)
            errors[m_index, s, z+1] = pred_signals[m_index, s, z+1]-signals[s][indexes[1]]

        if z == 1000:
            print("Index: ", end="")
        if z%1000 == 0:
            print(str(z), end="")
            sys.stdout.flush()
        if z % 250 == 0:
            print(".", end="")
            sys.stdout.flush()

        if indexes[1] > W:
            indexes[0] += 1
        indexes[1] += 1

        x.append(len(x)/fz)
        i += 1
    print(";")

np.savez(filename+".npz",
         x=x,
         pred_signals=pred_signals,
         errors=errors,
         signals_models=signals_models,
         signals_tests=signals_tests)
print(filename+".npz has been saved")

# plt.figure(A)
# plt.title("Physiological Signal Synthesizer", fontsize=32)
# plt.plot(x, X_train[0][0:indexes[1]],'g')
#
# for m_index in range(len(models)):
#     plt.plot(x, pred_signals[m_index,:], label=signals_models[m_index]["s_name"])
#
# plt.legend()
# plt.draw()
# plt.figure(B)

# for m_index in range(len(models)):
#     plt.plot(x, errors[m_index, :])
#     plt.plot(x, errors[m_index, :], label=signals_models[m_index]["s_name"])
#
# plt.legend()
# plt.show()
#
#

