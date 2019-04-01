import os
import numpy as np
import pandas as pd
import sys
import time

df = pd.read_csv('./data/_composite/zzz_composite.csv')

atbts = [x for x in list(df) if not 'interval' in x]

print(len(atbts))

atbts.remove('score')
atbts.remove('code_module')
atbts.remove('code_presentation')
atbts.remove('id_student')
atbts.remove('id_assessment')

print(len(atbts))

for item in atbts:
    print(item)