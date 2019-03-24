import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']
modules = ['aaa']

for mod in modules:

    mod_frame = pd.read_csv(f'./data/_composite/{mod}_composite.csv')

    drop_cols = ['code_module', 'code_presentation', 'id_student', 'id_assessment', 'score']

    y = mod_frame['score'].apply(lambda x: 'pass' if x >= 40 else 'fail')
    X = mod_frame.drop(columns=drop_cols)

    
