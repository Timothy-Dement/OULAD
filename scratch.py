import os
import numpy as np
import pandas as pd
import sys
import time

df = pd.read_csv('./data/_composite/zzz_composite.csv')

for item in df['highest_education'].unique():
    print(item)