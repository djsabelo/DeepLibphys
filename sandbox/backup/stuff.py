from DeepLibphys.utils.functions.common import *
import DeepLibphys.utils.functions.database as db
import DeepLibphys.models.LibphysSGDGRU_dev as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import numpy as np

signal = np.random.randint(0, 63, size=1)
model_info = db.ecg_1024_clean_models[6]
signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                                hidden_dim=model_info.Hd)
model = GRU.LibphysSGDGRU(signal2Model)

Z = model.synthesize_signal(4)