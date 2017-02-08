import BiosignalsDeepLibphys.utils.functions.libphys_GRU as GRU
from BiosignalsDeepLibphys.utils.functions.common import get_fantasia_dataset, get_signals
import matplotlib.pyplot as plt
import numpy as np
#CONFUSION_TENSOR_[W,Z]
Z = 200
W = 256
filename = "CONFUSION_TENSOR_["+str(Z)+","+str(W)+"]"

signals_models = [  {"Sd": 64, "Hd": 256, "name": "resp_[1.2.3.4.5.6.7.8.9.10]_old_fantasia", "dir": "RESP_FANTASIA[1000.256]",
                     "DS":-5, "t": -5, "W": 512, "s_name": "RESP"},
                    {"Sd":64, "Hd":256, "name":"ecg_old_fantasia", "dir":"FANTASIA[32x10.256]",
                     "DS":-5,"t":-5,"W":256,"s_name":"ECG"},
                    {"Sd": 64, "Hd": 128, "name": "provadeesforco_emg", "dir": "EMG","DS":480, "t":2000, "W":256,
                     "s_name": "EMG"},
                    {"Sd": 64, "Hd": 256, "name": "eeg_all", "dir": "EEG_Attention[1000.256]", "DS": -5, "t": -5,
                     "W": 256,
                     "s_name": "EEG"}]

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
                    ["other",'DRIVER_GSR',         0,  55800,  "GSR"],
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

loss_tensor = np.zeros((len(signals_models), len(signals_tests), Z))
n = -1
m = -1
z = -1
N_Signals = len(signals_tests)
random_indexes = np.zeros((len(signals_tests), Z))

X_train = []
Y_train = []
for i in range(len(signals_tests)):
    for j in range(Z):
        random_indexes[i, j] = np.random.randint(W, signals_tests[i][3] - W - 1)


s = -1;
history = [""]
X_matrix = np.zeros((N_Signals, W, Z))
Y_matrix = np.zeros((N_Signals, W, Z))
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
    print(N)
    for index in random_indexes[s, :]:
        z += 1
        X_matrix[s, :, z] = X_train[i][int(index):int(index + W)]
        Y_matrix[s, :, z] = Y_train[i][int(index):int(index + W)]


n = -1
for model_info in signals_models:
    n += 1
    s = -1
    model = GRU.LibPhys_GRU(model_info['Sd'], hidden_dim=model_info["Hd"], signal_name=model_info["name"])
    model.load(signal_name=model_info["name"], filetag=model.get_file_tag(model_info["DS"],
                                                                          model_info["t"]), dir_name=model_info["dir"])
    print("Processing " + model_info["name"])

    for signal_info in signals_tests:
        print("Calculating loss for " + signal_info[4], end=': ')
        s += 1
        z = -1
        print(N)
        for index in random_indexes[s, :]:
            z += 1
            loss_tensor[n, s, z] = model.calculate_loss([X_matrix[s, :, z]],[Y_matrix[s, :, z]])
            print(str(loss_tensor[n, s, z]), end=', ')
        print(";")



    # for signal_info in signals_tests:
    #     print("Calculating loss for " + signal_info[4], end=': ')
    #     m += 1
    #     z = -1
    #     X_train = []
    #     Y_train = []
    #     if signal_info[0] == "fantasia":
    #         X_train, Y_train = get_fantasia_dataset(model_info['Sd'], [signal_info[2]], signal_info[1],
    #                                                 peak_into_data=False)
    #     elif signal_info[0] == "emg":
    #         X_train, Y_train = get_signals(model_info['Sd'], signal_info[1],
    #                                                 peak_into_data=False, decimate=2, val='', row=6)
    #
    #     else:
    #         X_train, Y_train = get_signals(model_info['Sd'], signal_info[1],
    #                                                 peak_into_data=False)
    #
    #     N = len(X_train[0])
    #     print(N)
    #     for index in random_indexes[m, :]:
    #         z += 1
    #         loss_tensor[n, m, z] = model.calculate_loss([X_train[0][int(index):int(index + W)]],
    #                                                                    [Y_train[0][int(index):int(index + W)]])
    #         print(str(loss_tensor[n, m, z]), end=', ')
    #     print(";")

np.savez(filename+".npz",
         loss_tensor=loss_tensor,
         signals_models=signals_models,
         signals_tests=signals_tests)