import BiosignalsDeepLibphys.utils.functions.libphys_GRU_DEV as GRU
from BiosignalsDeepLibphys.utils.functions.common import get_fantasia_dataset, get_signals, segment_signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys

#CONFUSION_WINDOWS_[W,Z]
W = 256
Z = 100
filename = "CONFUSION_WINDOWS_["+str(W)+","+str(Z)+"]"

signals_models = [  {"Sd":64, "Hd":256, "name":"ecg_old_fantasia", "dir":"FANTASIA[32x10.256]",
                     "DS":0,"t":700,"W":256,"s_name":"ECG_NEW"},
                    # {"Sd": 64, "Hd": 256, "name": "resp_[1.2.3.4.5.6.7.8.9.10]_old_fantasia", "dir": "RESP_FANTASIA[1000.256]",
                    #  "DS":-5, "t": -5, "W": 256, "s_name": "RESP"},
                    # {"Sd":64, "Hd":256, "name":"ecg_[1.2.5.6.7.8.9.10]_old_fantasia","dir":"FANTASIA[1000.256]","DS":-5,
                    #  "t":-5,"W":256,"s_name":"ECG"},
                    # {"Sd": 64, "Hd": 256, "name": "eeg_all", "dir": "EEG_Attention[1000.256]","DS":-5, "t":-5, "W":256,
                    #  "s_name": "EEG"}
                    ]

#signals_models = [type (eeg, fantasia, emg or other), directory name, example, name for graph]
signals_tests = [ #["resp", 'Fantasia/RESP/mat/', 19, 900000, "RESP 7"],
                    ["ecg", 'Fantasia/ECG/mat/',  7,  900000, "ECG 7"],
                  ["ecg", 'Fantasia/ECG/mat/', 19,  900000, "ECG 19"],

                ]



N_Signals = len(signals_tests)
N_Models = len(signals_models)
X_train = []
Y_train = []

s = -1;
history = [""]
N_windows = 0
X_train, Y_train = get_fantasia_dataset(signals_models[0]['Sd'], [7], signals_tests[0][1],
                                        peak_into_data=False)

models = []

for model_info in signals_models:
    model = GRU.LibPhys_GRU(model_info['Sd'], hidden_dim=model_info["Hd"], signal_name=model_info["name"])
    model.load(signal_name=model_info["name"], filetag=model.get_file_tag(model_info["DS"],
                                                                          model_info["t"]), dir_name=model_info["dir"])
    models.append(model)

pred = models[0].generate_predicted_signal_2(X_train[0][0:512])
print(pred)

