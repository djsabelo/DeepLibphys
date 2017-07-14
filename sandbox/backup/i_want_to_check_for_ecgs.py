import numpy as np

from DeepLibphys.utils.functions.common import plot_confusion_matrix


def get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests, no_numbers=False):
    labels_model = np.asarray(np.zeros(len(Mod)*2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig)*2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1,len(Mod)*2,2))] = [signals_models[i]["s_name"] for i in Mod]
    labels_signals[list(range(1,len(Sig)*2,2))] = [signals_tests[i][-1] for i in Sig]

    predicted_matrix = np.argmin(loss_tensor[Mod][:, Sig, :], axis = 0)

    sinal_predicted_matrix = np.zeros((len(Sig), len(Mod)))
    N_Model = np.shape(sinal_predicted_matrix)[0]
    N_Signal = np.shape(sinal_predicted_matrix)[1]

    for i in range(N_Model):
        for j in range(N_Signal):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i,:] == j)

    return sinal_predicted_matrix, labels_model, labels_signals

def get_confusion_matrix(labels, signal_predicted_matrix):
    correct = 0
    wrong = 0
    N = 0
    models = list(range(np.shape(signal_predicted_matrix)[1]))
    confusion_tensor = np.zeros((len(models), 2, 2))
    rejection =  np.zeros(len(models))

    for i in range(np.shape(signal_predicted_matrix)[0]):
        values = signal_predicted_matrix[i, :]
        N += np.sum(values)
        correct += values[labels[i]]
        values = np.delete(values, labels[i])
        wrong += np.sum(values)
        #[TP,FN]
        #[FP,TN]
        for j in range(len(models)):
            values = signal_predicted_matrix[i, :]
            if labels[i] == j:
                confusion_tensor[j, 0, 0] += values[j]
                values = np.delete(values, j)
                confusion_tensor[j, 1, 0] += np.sum(values)
            else:
                confusion_tensor[j, 0, 1] += values[j]
                values = np.delete(values, j)
                confusion_tensor[j, 1, 1] += np.sum(values)

            rejection[j] += values[-1]

    return confusion_tensor, correct, wrong, rejection

def print_confusion(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=False):
    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)
    plot_confusion_matrix(sinal_predicted_matrix.T, labels_signals, labels_model, no_numbers)# , cmap=cmap)

def print_mean_loss(Mod, Sig, loss_tensor, signals_models, signals_tests):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i]["s_name"] for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i][-1] for i in Sig]

    mean_values_matrix = np.mean(loss_tensor, axis = 2)

    sinal_predicted_matrix = np.zeros(len(Sig))

    # for i in range(np.shape(sinal_predicted_matrix)[0]):
    for j in range(np.shape(sinal_predicted_matrix)[0]):
        sinal_predicted_matrix[j] = mean_values_matrix[0, j]

    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)
    plot_confusion_matrix(sinal_predicted_matrix.T, labels_model, labels_signals)  # , cmap=cmap)

# filename = "EEG_ECG_RESP_tensor"
# filename = "CONFUSION_TENSOR_[100,64]"  #wrong
# filename = "CONFUSION_TENSOR_[100,128]"  #wrong
# filename = "CONFUSION_TENSOR_[1000,256]" #wrong
filename =  "../data/validation/DNN_signal_detector" #wrong
# filename = "CONFUSION_TENSOR_[100,256]"

npzfile = np.load(filename+".npz")
loss_tensor, signals_models, signals_tests = \
    npzfile["loss_tensor"], npzfile["signals_models"], npzfile["signals_tests"]

# loss_tensor = [model, signal, experiment_number]
# predicted_matrix = [model, signal]


# PRINT MATRIX FOR SIGNALS OF DIFFERENT TYPES
# Mod = [0, 2, 3, 4]
# Sig = [2, 4, 0, 5]

# print_confusion(Mod, Sig, loss_tensor, signals_models, signals_tests)

# PRINT MATRIX FOR DIFERENT MODELS OF DIFFERENT TYPES
# Mod = [0,  3, 4]
# Sig = [1, 2, 3, 0, 5]

# PRINT MATRIX FOR DIFFERENT SOURCES
# Mod = [0,  1]
# Sig = [1, 2, 3]

# PRINT MATRIX FOR ALL SIGNALS
Mod = [0]
Sig = list(range(len(signals_tests)))
# print_mean_loss(Mod, Sig, loss_tensor, signals_models, signals_tests)

# signals_models = [  {"Sd": 64, "Hd": 256, "name": "resp_[1.2.3.4.5.6.7.8.9.10]_old_fantasia", "dir": "RESP_FANTASIA[1000.256]",
#                      "DS":-5, "t": -5, "W": 256, "s_name": "RESP"},
#                     {"Sd":64, "Hd":256, "name":"ecg_[1.2.5.6.7.8.9.10]_old_fantasia",
#                      "dir":"FANTASIA[1000.256]","DS":-5,"t":-5,"W":256,"s_name":"ECG"},
#                     {"Sd": 64, "Hd": 256, "name": "eeg_all", "dir": "EEG_Attention[1000.256]",
#                      "DS": -5, "t": -5, "W": 256, "s_name": "EEG"}
#                     ]
thresh = 300
Mod = list(range(np.shape(loss_tensor)[0]))
# Sig = list(range(np.shape(loss_tensor)[1]))
# Sig = list(range(1, 20))+list(range(21, np.shape(loss_tensor)[1]))
signals_models = np.hstack((signals_models, {"Sd":64, "Hd":256, "name":"no model", "dir":"FANTASIA[1000.256]","DS":-5,"t":-5,"W":256,"s_name":"None"}))
loss_tensor = np.vstack((loss_tensor, thresh*np.ones((1,np.size(loss_tensor, axis=1), np.size(loss_tensor, axis=2)))))


sinal_predicted_matrix, signal_labels, model_labels = get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests, no_numbers=True)
print_confusion(sinal_predicted_matrix, model_labels, signal_labels, no_numbers=True)


# confusion_tensor, trues, wrongs, rejection = get_confusion_matrix(label, sinal_predicted_matrix)
#
# print(trues)
# print(wrongs)
# print(confusion_tensor)
#
#
#
# plot_confusion_matrix_with_pie(confusion_tensor[0].T, ["RESP","NOT RESP"], ["PRED RESP","PRED NOT RESP"],rejection[0], title="RESP", no_numbers=True, cmap=plt.cm.Greens, cmap_text=plt.cm.Greens_r, norm=False)
# plot_confusion_matrix_with_pie(confusion_tensor[1].T, ["ECG","NOT ECG"], ["PRED RESP","PRED NOT RESP"], rejection[1], title="ECG", no_numbers=True, cmap=plt.cm.Reds, cmap_text=plt.cm.Reds_r, norm=False)
# plot_confusion_matrix_with_pie(confusion_tensor[2].T, ["ECG_NEW","NOT ECG"], ["PRED RESP","PRED NOT RESP"], rejection[1], title="ECG", no_numbers=True, cmap=plt.cm.Reds, cmap_text=plt.cm.Reds_r, norm=False)
# plot_confusion_matrix_with_pie(confusion_tensor[3].T, ["EEG","NOT EEG"], ["PRED RESP","PRED NOT RESP"], rejection[2], title="EEG", no_numbers=True, cmap=plt.cm.Blues, cmap_text=plt.cm.Blues_r, norm=False)
# plot_confusion_matrix_with_pie(confusion_tensor[4].T, ["NONE","NOT NONE"], ["PRED NONE","PRED NOT NONE"], title="NONE", no_numbers=True, cmap=plt.cm.Greys, cmap_text=plt.cm.Greys_r, norm=False)