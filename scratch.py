import os
import numpy as np
import pandas as pd
import sys
import time

for clf in ['dt', 'knn', 'nb', 'rf', 'svm']:

    df = pd.read_csv(f'./results/{clf}-results.csv')

    base_df = df[df['sample'] == 'non-smote']
    smote_df = df[df['sample'] == 'smote']

    base_df.to_csv(f'./results/{clf}_results.csv', index=False)
    smote_df.to_csv(f'./results/{clf}_smote_results.csv', index=False)