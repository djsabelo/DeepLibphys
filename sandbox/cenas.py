import DeepLibphys.models.LibphysMBGRU as MBGRU
import seaborn
from DeepLibphys.utils.functions.signal2model import *
from DeepLibphys.utils.functions.common import get_signals_tests
from DeepLibphys.utils.functions.database import *

signal2model = Signal2Model("XPTO", "XPTO")
signals = get_signals_tests(signal_tests, index=1)

model = MBGRU.LibphysMBGRU(signal2model)
model.train(signals[0], signal2model, loss_interval=10)


