import matplotlib.pyplot as plt
import numpy as np
from BiosignalsDeepLibphys.utils.functions.common import plot_confusion_matrix, make_cmap, get_color

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



npzfile = np.load("CONFUSION_TENSOR_[1000,256].npz")
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
Mod = [0, 1, 2, 3]
Sig = list(range(np.shape(loss_tensor)[1]))

print_confusion(Mod, Sig, loss_tensor, signals_models, signals_tests)