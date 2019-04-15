import os
import numpy as np
import pandas as pd
import sys
import time

df = pd.read_csv('./data/vle.csv')

types = df['activity_type'].sort_values().unique()

for t in types:
    print(t)