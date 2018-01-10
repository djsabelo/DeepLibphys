import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
import DeepLibphys.models.LibphysSGDGRU as GRU

model_info = db.ecg_1024_256_RAW[6]

model = GRU.LibphysSGDGRU(model_info.to_signal2model())

model.load(dir_name=model_info.directory)

model.online_sinthesizer(20000, [60], window_seen_by_GRU_size=512, uncertaintly=0.1)