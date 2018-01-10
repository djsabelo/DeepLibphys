import DeepLibphys.utils.functions.database as db
import pandas as pd
import DeepLibphys.models.LibphysMBGRU as GRU
from DeepLibphys.utils.functions.signal2model import Signal2Model
import numpy as np

def exists(where, what):
    try:
        where.index(what)
        return True
    except ValueError:
        return False

signals_models = db.ecg_1024_256_RAW + db.cybhi_512_M1 + db.cybhi_512_M2 + db.mit_1024
model_info = signals_models[0]

signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                            hidden_dim=model_info.Hd)

model = GRU.LibphysMBGRU(signal2Model)
times_in_hours = [[], [], [], [], []]
for model_info in signals_models:
    try:
        model.model_name = model_info.dataset_name
        dirx = model_info.directory
        model.load(dir_name=dirx)
        hours = model.train_time/(3.6 * 10**6)
        times_in_hours[0].append("cybhi" if exists(model.model_name, "cybhi") else "mit" if exists(model.model_name, "mit") else "fantasia")
        times_in_hours[1].append(256 if exists(dirx, "256") else 1024 if exists(dirx, "1024") else 12)
        times_in_hours[2].append(model_info.W)
        times_in_hours[3].append(hours)
        times_in_hours[4].append(model.model_name)
        print(hours)
    except:
        pass
print(np.array(times_in_hours).T)
df = pd.DataFrame(np.array(times_in_hours).T)
df.columns = ['Database', 'Batch Size', 'Window Size', 'Training Time in Hours', 'Model Name']
df.to_csv("times.csv")
print(df)