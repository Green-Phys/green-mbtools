# __init__.py for mbanalysis.data
from os import path as pathos
from pathlib import Path


data_path = Path(__file__).parent.absolute()

# IR grid files
ir_1e3 = pathos.join(data_path, '1e3_72.h5')
ir_1e4 = pathos.join(data_path, '1e4_104.h5')
ir_1e5 = pathos.join(data_path, '1e5_136.h5')
ir_1e6 = pathos.join(data_path, '1e6_168.h5')
ir_1e7 = pathos.join(data_path, '1e7_202.h5')