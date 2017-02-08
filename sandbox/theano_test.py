import numpy as np
import theano
import theano.tensor as T
import BiosignalsDeepLibphys.utils.functions.libphys_GRU_DEV as GRU
import BiosignalsDeepLibphys.utils.data.database as db
from BiosignalsDeepLibphys.utils.functions.common import get_signals_tests, segment_signal
from matplotlib import pyplot as plt
import seaborn

signals_models = db.signal_models
signals_tests  = db.signal_tests


def load_model(model_info, N_Windows):
    model = GRU.LibPhys_GRU(model_info.Sd, hidden_dim=model_info.Hd, signal_name=model_info.dataset_name, n_windows=N_Windows)
    model.load(signal_name=model_info.dataset_name, filetag=model.get_file_tag(model_info.DS,model_info.t),
               dir_name=model_info.directory)
    return model


def get_models(signal_models, index = None, N_Windows = None):
    models = []

    if index is None:
        for model_info in signals_models:
            models.append(load_model(model_info, N_Windows))
    else:
        model_info = signals_models[index]
        models.append(load_model(model_info, N_Windows))

    return models

N_Windows = 2048
signal_ecg = get_signals_tests(signals_tests, signals_models[0].Sd, index=7)[0][0]
signal_resp = get_signals_tests(signals_tests, signals_models[0].Sd, index=27)[0][0]
# predicted_signals = list(range(len(signals_tests)))
# model_errors = list(range(len(signals_tests)))
fi = np.random.randint(0, 50000)
W = 256
[ecg_segments_, y_ecg, _a, end_index] = segment_signal(signal_ecg, W, 0, N_Windows, start_index=fi)
[resp_segments, y_resp, _a, end_index] = segment_signal(signal_resp, W, 0, N_Windows, start_index=fi)
model = get_models(signals_models, index=0, N_Windows=N_Windows)[0]


# [pred_signal_ecg,error] = model.predict_class(ecg_segments, y)
[pred_signal_ecg, e_ecg] = model.predict_class(np.asarray(ecg_segments_, dtype=int), np.asarray(y_ecg, dtype=int))
[pred_signal_resp, e_resp] = model.predict_class(resp_segments, y_resp)
print(e_ecg)
index = np.argmin(np.std(e_resp, axis=1))
plt.plot(e_ecg[0], label=signals_tests[7].name)
plt.plot(e_resp[0], label=signals_tests[27].name)
# plt.plot(e_ecg[-1], label=signals_tests[7].name+" last")
# plt.plot(e_resp[-1], label=signals_tests[27].name+" last")

plt.figure()
plt.plot(pred_signal_ecg[0], label=signals_tests[7].name)
plt.plot(pred_signal_resp[0], label=signals_tests[27].name)
# plt.plot(pred_signal_ecg[-1], label=signals_tests[3].name+" last")
# plt.plot(pred_signal_resp[-1], label=signals_tests[27].name+" last")
plt.plot(signal_ecg[fi+0+1:fi+0+1+len(pred_signal_ecg[0,:])], label="Original_signal")

plt.legend()
plt.show()
# signal = model.generate_predicted_signal(1, np.asarray([[1,2,3,4],[1,2,3,4]]), 256)
# print(signal)