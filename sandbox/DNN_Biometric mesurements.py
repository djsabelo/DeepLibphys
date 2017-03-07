import matplotlib.pyplot as plt
import numpy as np
import seaborn

import sys
sys.path.append("/home/belo/PycharmProjects/BiosignalsLibphysGroup")
print(sys.path)

import DeepLibphys.models.libphys_MBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import (get_random_batch,
                                                get_signals_tests,
                                                plot_confusion_matrix,
                                                randomize_batch)
from DeepLibphys.utils.functions.database import ModelInfo

GRU_DATA_DIRECTORY = "../data/trained/"

def load_test_data(filetag=None, dir_name=None):
    print("Loading test data...")
    filename = GRU_DATA_DIRECTORY + dir_name + '/' + filetag + '_test_data.npz'
    npzfile = np.load(filename)
    return npzfile["test_data"]

def calculate_loss_tensors(N_Windows, W, signals_models):
    N_Versions = len(signals_models)
    N_Signals = len(signals_models[0])
    loss_tensor = np.zeros((N_Versions, N_Signals, N_Signals, N_Windows))
    X_matrix = np.zeros((N_Versions, N_Signals, N_Windows, W))
    Y_matrix = np.zeros((N_Versions, N_Signals, N_Windows, W))

    i = 0
    for model_info in signals_models[0]:
        x_tests = []
        y_tests = []
        for version in range(N_Versions):
            [x_test, y_test] = load_test_data(
                "GRU_" + model_info.dataset_name + "[" + str(model_info.Sd) + "." + str(model_info.Hd) + ".-1.-1.-1]"
                , model_info.directory)
            x_tests.append(x_test)
            y_tests.append(y_test)
        X_matrix[:, i, :, :], Y_matrix[:, i, :, :] = randomize_batch(np.asarray(x_test), np.asarray(y_test), N_Windows)
    i += 1

    print("Loading base model...")
    model_info = signals_models[0][0]
    model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
                            n_windows=N_Windows)

    for m in range(N_Signals):
        for version in range(N_Versions):
            model_info = signals_models[version][m]
            model.signal_name = model_info.dataset_name
            model.load(signal_name=model_info.name, filetag=model.get_file_tag(model_info.DS,
                                                                               model_info.t),
                       dir_name=model_info.directory)
            print("Processing " + model_info.name)

            for s in range(N_Signals):
                x_test = X_matrix[version, s, :, :]
                y_test = Y_matrix[version, s, :, :]
                print("Calculating loss for " + signals_models[version][s].name, end=';\n ')
                loss_tensor[version, m, s, :] = np.asarray(model.calculate_loss_vector(x_test, y_test))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models,
             signals_tests=signals_info)

    return loss_tensor

def calculate_loss_tensor(filename, N_Windows, W, signals_models, signals_info, test_included=False):
    loss_tensor = np.zeros((len(signals_models), len(signals_info), N_Windows))
    N_Signals = len(signals_info)

    X_matrix = np.zeros((N_Signals, N_Windows, W))
    Y_matrix = np.zeros((N_Signals, N_Windows, W))


    if not test_included:
        signals = get_signals_tests(signals_info, signals_models[0].Sd)
        for i in range(N_Signals):
            [X_matrix[i, :, :], Y_matrix[i, :, :]] = get_random_batch(signals[0][i], signals[1][i], W, N_Windows)
    else:
        i = 0
        for model_info in signals_models:
            [x_test, y_test] = load_test_data("GRU_" + model_info.dataset_name + "["+str(model_info.Sd)+"."+str(model_info.Hd)+".-1.-1.-1]"
                                              , model_info.directory)
            X_matrix[i, :, :], Y_matrix[i, :, :] = randomize_batch(x_test, y_test, N_Windows)
            i += 1

    print("Loading model...")
    model_info = signals_models[0]
    model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
                            n_windows=N_Windows)
    history = []
    m = -1
    for m in range(len(signals_models)):
        model_info = signals_models[m]
        model.signal_name = model_info.dataset_name
        model.load(signal_name=model_info.name, filetag=model.get_file_tag(model_info.DS,
                                                                           model_info.t),
                   dir_name=model_info.directory)
        print("Processing " + model_info.name)

        for s in range(N_Signals):
            x_test = X_matrix[s, :, :]
            y_test = Y_matrix[s, :, :]
            signal_info = signals_info[s]
            print("Calculating loss for " + signal_info.name, end=';\n ')
            loss_tensor[m, s, :] = np.asarray(model.calculate_loss_vector(x_test, y_test))

    np.savez(filename + ".npz",
             loss_tensor=loss_tensor,
             signals_models=signals_models,
             signals_tests=signals_info)

    return loss_tensor


def get_sinal_predicted_matrix(Mod, Sig, loss_tensor, signals_models, signals_tests, N_Windows, no_numbers=False):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i].name for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i].name for i in Sig]

    for i in Sig:
        # if signals_models[i].name is not "None":
        loss_tensor[:-1, i, :] = loss_tensor[:-1, i, :] / np.max(loss_tensor[:-1, i, :])

    predicted_matrix = np.argmin(loss_tensor[Mod][:, Sig, :], axis=0)

    sinal_predicted_matrix = np.zeros((len(Sig), len(Mod)))

    for i in range(np.shape(sinal_predicted_matrix)[0]):
        for j in range(np.shape(sinal_predicted_matrix)[1]):
            sinal_predicted_matrix[i, j] = sum(predicted_matrix[i, :] == j) / N_Windows

    return sinal_predicted_matrix, labels_model, labels_signals

def calculate_prediction_matrix(loss_tensor, threshold = 1):
    threshold_layer = np.ones((np.shape(loss_tensor)[1],np.shape(loss_tensor)[2])) * threshold
    normalized_loss_tensor = np.zeros((np.shape(loss_tensor)[0]+1,np.shape(loss_tensor)[1],np.shape(loss_tensor)[2]))
    predicted_tensor = np.zeros_like(loss_tensor)

    for j in range(np.shape(loss_tensor)[1]):
        normalized_loss_tensor[:-1,j,:] = loss_tensor[:,j,:] / np.max(loss_tensor[:,j,:], axis=0)

    # normalized_loss_tensor[:-1,:,:] = loss_tensor
    normalized_loss_tensor[-1, :, :] = threshold_layer
    # print("\nThreshold "+str(threshold), end=":")

    # plt.figure("LOSS NORMALIZED")
    # plt.plot(normalized_loss_tensor[:, 1, 0])
    # plt.plot(normalized_loss_tensor[:, 1, 0])
    # plt.show()

    for i in range(np.shape(loss_tensor)[1]):
        predicted_tensor[i, : , :] = np.argmin(normalized_loss_tensor[[-1, i], :, :], axis=0)

    predicted_matrix = np.argmin(normalized_loss_tensor, axis=0)

    return predicted_matrix, predicted_tensor


# def calculate_new_prediction_matrix(loss_tensor, threshold=1):
#     N_predicted, N_signals, N_Windows = np.shape(loss_tensor)
#
#     threshold_layer = np.ones((np.shape(loss_tensor)[1],np.shape(loss_tensor)[2])) * threshold
#     normalized_loss_tensor = np.zeros((np.shape(loss_tensor)[0]+1,np.shape(loss_tensor)[1],np.shape(loss_tensor)[2]))
#
#
#     for j in range(np.shape(loss_tensor)[1]):
#         normalized_loss_tensor[:-1, j, :] = loss_tensor[:, j, :] /np.max(loss_tensor[:, j, :], axis=0)
#
#     predicted_matrix = np.zeros((N_signals,N_Windows)) - 1
#     normalized_loss_tensor[-1, :, :] = threshold_layer
#     for i in range(np.shape(normalized_loss_tensor)[0]-1):
#         print("MIN LOSS TENSOR")
#         selected = np.argmin(normalized_loss_tensor[[-1, i],:,:], axis=0)
#         predicted_matrix[i, np.where(selected[i, :] == 0)[0]] = np.shape(normalized_loss_tensor)[0]-1
#         predicted_matrix[i, np.where(selected[i, :] == 1)[0]] = i
#
#
#
#
#     return predicted_matrix



def get_confusion_matrix(labels, signal_predicted_matrix):
    correct = 0
    wrong = 0
    N = 0
    models = list(range(np.shape(signal_predicted_matrix)[1]))
    confusion_tensor = np.zeros((len(models), 2, 2))
    rejection = np.zeros(len(models))
    N_Windows = np.sum(signal_predicted_matrix[:, 0])

    for i in range(np.shape(signal_predicted_matrix)[0]):
        values = signal_predicted_matrix[i, :]
        N += np.sum(values)
        correct += values[labels[i]]
        values = np.delete(values, labels[i])
        wrong += np.sum(values)
        # [TP,FN]
        # [FP,TN]
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


def get_signal_confusion_matrix(signal_predicted_matrix):
    signal_list = list(range(np.shape(signal_predicted_matrix)[0]))
    confusion_tensor = np.zeros((len(signal_list), 2, 2))
    N_Windows = np.sum(signal_predicted_matrix[:, 0])
    rejection_rate = np.zeros_like(signal_list)
    rejection_signal = np.shape(signal_predicted_matrix)[0]

    for signal in signal_list:
        # [TP,FN]
        # [FP,TN]
        confusion_tensor[signal, 0, 0] = len(np.where(signal_predicted_matrix[signal, :] == signal)[0]) # TP
        confusion_tensor[signal, 0, 1] = len(np.where(signal_predicted_matrix[signal, :] != signal)[0]) # FN

        confusion_tensor[signal, 1, 0] = len(np.where(signal_predicted_matrix[np.arange(np.shape(signal_predicted_matrix)[0]) != signal, :] == signal)[0]) # FP
        confusion_tensor[signal, 1, 1] = len(np.where(signal_predicted_matrix[np.arange(np.shape(signal_predicted_matrix)[0]) != signal, :] != signal)[0]) # TN

        rejection_rate[signal] = len(np.where(signal_predicted_matrix[signal, :] == rejection_signal)[0]) / N_Windows

    return confusion_tensor, rejection_rate

def get_classification_confusion(signal_predicted_tensor):
    signal_list = np.arange(np.shape(signal_predicted_tensor)[0])
    confusion_tensor = np.zeros((len(signal_list), 2, 2))
    N_Windows = np.sum(signal_predicted_tensor[:, 0])
    rejection_rate = np.zeros_like(signal_list)
    rejection_signal = np.shape(signal_predicted_tensor)[0]

    for signal in signal_list:
        # [TP,FN]
        # [FP,TN]
        classified_matrix = signal_predicted_tensor[signal, :, :]
        # plt.plot(classified_matrix)
        # plt.show()

        confusion_tensor[signal, 0, 0] = len(np.where(classified_matrix[signal, :] == 1)[0]) # TP
        confusion_tensor[signal, 0, 1] = len(np.where(classified_matrix[signal, :] == 0)[0]) # FN

        confusion_tensor[signal, 1, 0] = len(np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 1)[0]) # FP
        confusion_tensor[signal, 1, 1] = len(np.where(np.squeeze(classified_matrix[np.where(signal_list != signal), :]) == 0)[0]) # TN

    # print(confusion_tensor[0,:,:])
    return confusion_tensor


def print_confusion(sinal_predicted_matrix, labels_signals, labels_model, no_numbers=False):
    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)


def print_mean_loss(Mod, Sig, loss_tensor, signals_models, signals_tests):
    labels_model = np.asarray(np.zeros(len(Mod) * 2, dtype=np.str), dtype=np.object)
    labels_signals = np.asarray(np.zeros(len(Sig) * 2, dtype=np.str), dtype=np.object)
    labels_model[list(range(1, len(Mod) * 2, 2))] = [signals_models[i]["s_name"] for i in Mod]
    labels_signals[list(range(1, len(Sig) * 2, 2))] = [signals_tests[i][-1] for i in Sig]

    mean_values_matrix = np.mean(loss_tensor, axis=2)

    sinal_predicted_matrix = np.zeros(len(Sig))

    # for i in range(np.shape(sinal_predicted_matrix)[0]):
    for j in range(np.shape(sinal_predicted_matrix)[0]):
        sinal_predicted_matrix[j] = mean_values_matrix[0, j]

    print(sinal_predicted_matrix)
    # cmap = make_cmap(get_color(), max_colors=1000)
    plot_confusion_matrix(sinal_predicted_matrix.T, labels_model, labels_signals)  # , cmap=cmap)


def classify_biosignals(filename, threshold=300, models_index=None, signals_index=None):
    npzfile = np.load(filename + ".npz")
    loss_tensor, signals_models, signals_tests = \
        npzfile["loss_tensor"], npzfile["signals_models"], npzfile["signals_tests"]

    if models_index is None:
        models_index = list(range(np.shape(loss_tensor)[0]))
    if signals_index is None:
        signals_index = list(range(np.shape(loss_tensor)[1]))

    signals_models = np.hstack((signals_models, ModelInfo(name="None", dataset_name="none")))

    loss_tensor = np.vstack(
        (loss_tensor, threshold * np.ones((1, np.size(loss_tensor, axis=1), np.size(loss_tensor, axis=2)))))

    sinal_predicted_matrix, signal_labels, model_labels = get_sinal_predicted_matrix(
        models_index, signals_index, loss_tensor, signals_models, signals_tests, N_Windows, no_numbers=True)

    print_confusion(sinal_predicted_matrix, model_labels, signal_labels, no_numbers=True)


def calculate_variables(loss_tensor, threshold=0.1):
    prediction_matrix, prediction_tensor = calculate_prediction_matrix(loss_tensor, threshold)
    N_Signals = np.shape(prediction_matrix)[0]
    N_Windows = np.shape(prediction_matrix)[1]
    probabilities = np.zeros_like(prediction_matrix)

    # Probability distribution
    # for i in range(N_Signals):
    #     for j in range(N_Signals):
    #         # probability of i being j
    #         probabilities[i,j] = (len(np.where(prediction_matrix[i,:] == j)[0]) / N_Windows)
    #         print(prediction_matrix[i,:])
    #
    #     plt.figure("Signal #"+str(i))
    #     plt.hist(prediction_matrix[i,:])
    #
    # plt.show()

    #

    confusion_tensor = get_classification_confusion(prediction_tensor)
    N_signals = np.shape(prediction_matrix)[0]
    labels = list(range(N_signals))

    # i = 0
    # label = labels[i]
    # prediction = prediction_matrix[:,i]
    # print(labels)
    # print(confusion_tensor[0,:,:])
    scores = list(range(N_Signals))

    # [TP,FN]
    # [FP,TN]
    for i in list(range(N_Signals)):
        # TRUE POSITIVE
        TP = confusion_tensor[i, 0, 0]
        # TRUE NEGATIVE
        TN = confusion_tensor[i, 1, 1]
        # FALSE POSITIVE
        FP = confusion_tensor[i, 1, 0]
        # FALSE NEGATIVE
        FN = confusion_tensor[i, 0, 1]

        # SENSITIVITY - TRUE POSITIVE RATE
        TPR = TP / (TP + FN)

        # SPECIFICITY - TRUE NEGATIVE RATE
        TNR = TN / (TN + FP)

        # PRECISION - POSITIVE PREDICTIVE VALUE
        PPV = TP / (TP + FP)

        # NEGATIVE PREDICTIVE VALUE
        NPV = TN / (TN + FN)

        # FALL-OUT - FALSE POSITIVE RATE
        FPR = FP / (FP + TN)

        # FALSE DISCOVERY RATE
        FDR = FP / (FP + TP)

        # MISS RATE - FALSE NEGATIVE RATE
        FNR = FN / (FN + TP)

        # ACCURACY
        ACC = (TP + TN) / (TP + TN + FP + FN)

        # F1 SCORE
        F1 = 2*TP / (2*TP + FP + FN)

        scores[i] = {  "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN":FN,
                    "TPR":TPR,
                    "TNR":TNR,
                    "PPV":PPV,
                    "NPV":NPV,
                    "FPR":FPR,
                    "FDR":FDR,
                    "FNR":FNR,
                    "ACC":ACC,
                    "F1":F1}

        # print(scores[i]["TPR"])
        # if i ==0:
        #     print(confusion_tensor[i])
        #     print("FPR - "+str(scores[i]["TPR"]))
        #     print("FNR - "+str(scores[i]["FNR"]))


    return scores


# CONFUSION_TENSOR_[W,Z]
N_Windows = 150
W = 256
filename = "../data/validation/CONFUSION_acc_biometric_ALL_[" + str(N_Windows) + "," + str(W) + "]"

signals_models = [db.biometry_x_models[:10], db.biometry_y_models[:10], db.biometry_z_models[:10]]
signals_info = db.signal_tests

if type is not None:
    signals_aux = []
    for signal_info in signals_info:
        if signal_info.type == "biometric":
            signals_aux.append(signal_info)
    signals_info = signals_aux

signals_info = signals_info[:10]
losses = np.zeros((3, len(signals_models[0]), len(signals_info), N_Windows))
losses = calculate_loss_tensors(N_Windows, W, signals_models)
loss_tensor = losses

npzfile = np.load(filename + ".npz")
loss_tensor, signals_models, signals_info = \
    npzfile["loss_tensor"], npzfile["signals_models"], npzfile["signals_tests"]

min_loss_tensor = np.min(losses, axis=0)
max_loss_tensor = np.max(losses, axis=0)
mean_loss_tensor = np.mean(losses, axis=0)

loss_tensor = losses[0]

# loss_tensor = mean_loss_tensor
step = 0.0001
N_Signals = np.shape(loss_tensor)[0]
eer = np.zeros((2,N_Signals)) - 1
x = np.arange(0, 1+step, step)
roc1 = np.zeros((2, len(x), np.shape(loss_tensor)[0]))
roc2 = np.zeros((2, len(x), np.shape(loss_tensor)[0]))
for i in range(len(x)):
    scores = calculate_variables(loss_tensor, x[i])
    for j in range(np.shape(loss_tensor)[0]):
        roc1[0, i, j] = scores[j]["FNR"]
        roc1[1, i, j] = scores[j]["FPR"]

        roc2[0, i, j] = scores[j]["TPR"]
        roc2[1, i, j] = scores[j]["FPR"]
    # print("ACC {0} - "+str(scores[0]["ACC"]))


name = 'gnuplot'
cmap = plt.get_cmap(name)
cmap_list = [cmap(i) for i in np.linspace(0, 1, N_Signals)]
# plt.figure("Signal #"+str(sig))
fig_1 = "ROC False Negative Rate/False Positive Rate"
fig_2 = "ROC True Positive Rate/False Positive Rate"
for j in range(np.shape(loss_tensor)[0]):
    try:
        eer_index = np.argmin(abs(roc1[0,:,j]-roc1[1,:,j]))
        eer[0, j] = roc1[0, eer_index, j]
        eer[1, j] = roc1[1, eer_index, j]
    except:
        pass

    plt.figure(fig_1)
    plt.scatter(roc1[1,:,j],roc1[0,:,j], marker='.', color=cmap_list[j], label='Signal #{0}'.format(j))

    plt.figure(fig_2)
    plt.scatter(roc2[1,:,j],roc2[0,:,j], marker='.', color=cmap_list[j], label='Signal #{0}'.format(j))

for signal in range(N_Signals):
    plt.figure(fig_1)
    plt.plot(eer[1,signal],eer[0, signal], color="#990000", marker='o', alpha=0.2)


    # plt.plot(roc1[1,eer[1,signal],signal], eer[0,signal],)

print(eer[0])
print(np.mean(eer[0, :]), end=" - ")
print(np.mean(eer[1, :]))
print(np.std(eer[0, :]), end=" - ")
print(np.std(eer[1, :]))


plt.figure(fig_1)
plt.plot([0,1],[0,1],  color="#990000", alpha=0.2)

plt.ylabel("False Negative Rate")
plt.xlabel("False Positive Rate")
plt.ylim([0,1])
plt.xlim([0,1])
plt.legend()

plt.figure(fig_2)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.ylim([0,1])
plt.xlim([0,1])
plt.legend()
plt.show()

models = list(range(len(signals_models)))
signals = list(range(len(signals_info)))
# classify_biosignals(filename, models_index=models, signals_index=signals[0], threshold=0.5)

# signals_models = db.signal_generic_models
# signals_info = db.signal_tests
#
# # if type is not None:
# #     signals_aux = []
# #     for signal_info in signals_info:
# #         if signal_info.type == "ecg":
# #             signals_aux.append(signal_info)
# #     signals_info = signals_aux
#
# calculate_loss_tensor(filename, N_Windows, W, signals_models, signals_info, type="ecg")
# models = list(range(len(signals_models)))
# signals = list(range(len(signals_info)))
# classify_biosignals(filename, models_index=models, signals_index=signals, threshold=0.08)
