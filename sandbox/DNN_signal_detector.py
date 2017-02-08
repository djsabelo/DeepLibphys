import numpy as np
import theano
import theano.tensor as T
import BiosignalsDeepLibphys.utils.functions.libphys_GRU_DEV as GRU
import BiosignalsDeepLibphys.utils.data.database as db
from BiosignalsDeepLibphys.utils.functions.common import get_signals_tests, segment_signal

import seaborn

filename = "../data/validation/ERRORs_for_all"
def process_error_by_prediction(filename):
    signals_models = db.signal_models
    signals_tests = db.signal_tests

    def load_model(model_info, N_Windows):
        model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name,
                                n_windows=N_Windows)
        model.load(signal_name=model_info.dataset_name, filetag=model.get_file_tag(model_info.DS, model_info.t),
                   dir_name=model_info.directory)
        return model

    def get_models(signal_models, N_Windows=None, index=None):
        models = []

        if index is None:
            for model_info in signals_models:
                models.append(load_model(model_info,N_Windows))
        else:
            model_info = signals_models[index]
            models.append(load_model(model_info,N_Windows))

        return models

    signals = get_signals_tests(signals_tests, signals_models[0].Sd)

    W = 256
    N_Windows = 20000
    for m in range(len(signals_models)):
        models = []
        models = get_models(signals_models, N_Windows, index=m)
        predicted_signals = list(range(len(signals_tests)))
        model_errors = list(range(len(signals_tests)))
        predicted_signals_ = list(range(len(signals_tests)))
        model_errors_ = list(range(len(signals_tests)))
        print("\nProcessing Model " + signals_models[m].name + ":")
        for s in range(len(signals)):
            print("\tProcessing Signal " + signals_tests[s].name + ";")
            signal = signals[s]
            if model_errors[s].__class__ is int:
                model_errors[s] = []
                model_errors_[s] = []
                predicted_signals_[s] = []
                predicted_signals[s] = []

            [segmented, y, N_Windows, last_index] = segment_signal(signal, W, 0, N_Windows)
            [x, e] = models[0].predict_class(segmented, y)
            predicted_signals[s].append(x[0,:])
            predicted_signals_[s].append(x[-1, :])
            model_errors[s].append(e[0,:])
            model_errors_[s].append(e[-1, :])
            limit = last_index + (N_Windows + W)
            print("processing...", end =" ")
            while limit < signals_tests[s].size:
                print(str(limit) + " of " + str(signals_tests[s].size), end="_")
                [segmented, y, N_Windows, last_index] = segment_signal(signal, W, 0, N_Windows, start_index=last_index)
                [x, e] = models[0].predict_class(segmented, y)
                predicted_signals[s][-1] = np.append(predicted_signals[s][-1], x[0,:])
                predicted_signals[s][-1] = np.append(predicted_signals_[s][-1], x[-1, :])
                model_errors[s][-1] = np.append(model_errors[s][-1], e[-1, :])
                model_errors_[s][-1] = np.append(model_errors_[s][-1], e[-1, :])
                # print(np.shape(predicted_signals[s][-1]))
                limit = last_index + (N_Windows + W)

        np.savez(filename + str(m) +".npz",
                 predicted_signals=predicted_signals,
                 model_errors=model_errors,
                 predicted_signals_=predicted_signals,
                 model_errors_=model_errors,
                 signals_models=signals_models,
                 signals_tests=signals_tests)
        print(filename + ".npz has been saved")


process_error_by_prediction(filename)
signals_models = db.signal_models
signals_tests  = db.signal_tests





#
# npzfile = np.load(filename+".npz")
# predicted_signals, model_errors, predicted_signals_, model_errors_, signals_models, signals_tests = \
#     npzfile["predicted_signals"], \
#     npzfile["model_errors"], \
#     npzfile["predicted_signals_"], \
#     npzfile["model_errors_"], \
#     npzfile["signals_models"], npzfile["signals_tests"]
#
#
# # signals = get_signals_tests(signals_tests, signals_models[0].Sd)
#
# Z = 100
# W = 256
# N_Signals = len(signals_tests)
# N_Models = len(signals_models)
# random_indexes = np.zeros((N_Signals, Z))
# loss_tensor_std = np.zeros((N_Models, N_Signals, Z))
# loss_tensor_mean = np.zeros((N_Models, N_Signals, Z))
# loss_tensor_quadratic_error = np.zeros((N_Models, N_Signals, Z))
# loss_tensor_std_ = np.zeros((N_Models, N_Signals, Z))
# loss_tensor_mean_ = np.zeros((N_Models, N_Signals, Z))
# loss_tensor_quadratic_error_ = np.zeros((N_Models, N_Signals, Z))
#
# for s in range(N_Signals):
#     random_indexes = np.random.randint(W, len(model_errors[s][0]) - W - 1, Z)
#     for z in range(Z):
#         for m in range(N_Models):
#             index = random_indexes[z]
#             loss_tensor_std[m, s, z] = np.std(model_errors[s][m][index:index+W])
#             loss_tensor_mean[m, s, z] = np.mean(model_errors[s][m][index:index + W])
#             loss_tensor_quadratic_error[m, s, z] = np.mean(model_errors[s][m][index:index + W]**2)
#             loss_tensor_std_[m, s, z] = np.std(model_errors_[s][m][index:index+W])
#             loss_tensor_mean_[m, s, z] = np.mean(model_errors_[s][m][index:index + W])
#             loss_tensor_quadratic_error_[m, s, z] = np.mean(model_errors_[s][m][index:index + W]**2)
#
# filename = "../data/validation/DNN_signal_detector"
# np.savez(filename + ".npz",
#          loss_tensor_std=loss_tensor_std,
#          loss_tensor_mean=loss_tensor_mean,
#          loss_tensor_quadratic_error=loss_tensor_quadratic_error,
#          loss_tensor_std_=loss_tensor_std_,
#          loss_tensor_mean_=loss_tensor_mean_,
#          loss_tensor_quadratic_error_=loss_tensor_quadratic_error_,
#          signals_models=signals_models,
#          signals_tests=signals_tests)
# print(filename + ".npz has been saved")
