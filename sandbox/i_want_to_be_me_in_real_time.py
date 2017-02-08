import BiosignalsDeepLibphys.utils.functions.libphys_GRU as GRU
from BiosignalsDeepLibphys.utils.functions.common import get_fantasia_dataset, get_signals, segment_signal
import matplotlib.pyplot as plt
import numpy as np
import sys

#CONFUSION_WINDOWS_[W,Z]
W = 256
Z = 900
filename = "CONFUSION_WINDOWS_["+str(W)+","+str(Z)+"]"

signals_models = [  {"Sd":64, "Hd":256, "name":"ecg_old_fantasia", "dir":"FANTASIA[32x10.256]",
                     "DS":0,"t":300,"W":512,"s_name":"ECG_NEW"},
                    {"Sd": 64, "Hd": 256, "name": "resp_[1.2.3.4.5.6.7.8.9.10]_old_fantasia", "dir": "RESP_FANTASIA[1000.256]",
                     "DS":-5, "t": -5, "W": 512, "s_name": "RESP"},
                    {"Sd":64, "Hd":256, "name":"ecg_[1.2.5.6.7.8.9.10]_old_fantasia","dir":"FANTASIA[1000.256]","DS":-5,
                     "t":-5,"W":512,"s_name":"ECG"},
                    {"Sd": 64, "Hd": 256, "name": "eeg_all", "dir": "EEG_Attention[1000.256]","DS":-5, "t":-5, "W":512,
                     "s_name": "EEG"}
                    ]

#signals_models = [type (eeg, fantasia, emg or other), directory name, example, name for graph]
signals_tests = [ ["ecg", 'Fantasia/ECG/mat/',  7,  900000, "ECG 7"],
                  ["ecg", 'Fantasia/ECG/mat/', 19,  900000, "ECG 19"]
                ]



N_Signals = len(signals_tests)
N_Models = len(signals_models)
X_train = []
Y_train = []

s = -1;
history = [""]
N_windows = 0
X_loss_window_tensor = []
Y_loss_window_tensor = []
for signal_info in signals_tests:
    print("Processing " + signal_info[4], end=': ')
    s += 1
    z = -1
    i = 0

    if signal_info[0] == "ecg":
        if history[-1] == "ecg":
            print("Processing ecg " + str(signal_info[2]-1), end=': ')
            i = signal_info[2]-1
        else:
            X_train, Y_train = get_fantasia_dataset(signals_models[0]['Sd'], list(range(1, 20)), signal_info[1],
                                                peak_into_data=False)
    elif signal_info[0] == "resp":
        if history[-1] == "resp":
            i = signal_info[2] - 1
        else:
            X_train, Y_train = get_fantasia_dataset(signals_models[0]['Sd'], list(range(1, 20)), signal_info[1],
                                                    peak_into_data=False)
    elif signal_info[0] == "emg":
        X_train, Y_train = get_signals(signals_models[0]['Sd'], signal_info[1],
                                       peak_into_data=False, decimate=2, val='', row=6)
    elif signal_info[0] == "eeg":
        if history[-1] == "eeg":
            i = signal_info[2]
        else:
            X_train, Y_train = get_signals(signals_models[0]['Sd'], signal_info[1],
                                       peak_into_data=False)

    else:
        X_train, Y_train = get_signals(signals_models[0]['Sd'], signal_info[1],
                                       peak_into_data=False)

    history.append(signal_info[0])
    N = len(X_train[i])

    [X_segment_matrix, N_windows] = segment_signal(X_train[i], W)
    [Y_segment_matrix, N_windows] = segment_signal(Y_train[i], W)

    X_segment_matrix = np.reshape(X_segment_matrix, (1, np.shape(X_segment_matrix)[0], np.shape(X_segment_matrix)[1]))
    Y_segment_matrix = np.reshape(Y_segment_matrix, (1, np.shape(Y_segment_matrix)[0], np.shape(Y_segment_matrix)[1]))

    if len(X_loss_window_tensor) == 0:
        X_loss_window_tensor = X_segment_matrix
        Y_loss_window_tensor = Y_segment_matrix
    else:
        X_loss_window_tensor = np.append(X_loss_window_tensor, X_segment_matrix, axis=0)
        Y_loss_window_tensor = np.append(Y_loss_window_tensor, Y_segment_matrix, axis=0)

    print(N)

u = 32*2
v = 32*2+Z
X_loss_window_tensor = X_loss_window_tensor[:,u:v,:]
Y_loss_window_tensor = Y_loss_window_tensor[:,u:v,:]
loss_tensor = np.zeros((N_Models, N_Signals, Z))
m = -1
for model_info in signals_models:
    m += 1
    s = -1
    model = GRU.LibPhys_GRU(model_info['Sd'], hidden_dim=model_info["Hd"], signal_name=model_info["name"])
    model.load(signal_name=model_info["name"], filetag=model.get_file_tag(model_info["DS"],
                                                                          model_info["t"]), dir_name=model_info["dir"])
    print("Processing " + model_info["name"])

    for signal_info in signals_tests:
        print("Calculating loss for " + signal_info[4], end='; ')
        s += 1
        print(Z)
        for w in range(Z):
            loss_tensor[m, s, w] = model.calculate_loss([X_loss_window_tensor[s, w, :]], [Y_loss_window_tensor[s, w, :]])
            print(str(loss_tensor[m, s, w]), end=', ')
            sys.stdout.flush()
        print(";")


np.savez(filename+".npz",
         loss_tensor=loss_tensor,
         signals_models=signals_models,
         signals_tests=signals_tests)
print(filename+".npz has been saved")