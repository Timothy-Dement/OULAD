import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_file = './dt-results.csv'

df = pd.read_csv(results_file)

samples = ['non-smote', 'smote']

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']

attributes = ['asmt', 'asmt_stdnt', 'asmt_abd', 'asmt_abi', 'asmt_stdnt_abd', 'asmt_stdnt_abi']

for smpl in samples:
    smpl_df = df[df['sample'] == smpl]
    for mod in modules:
        mod_df = smpl_df[smpl_df['module'] == mod]
        

