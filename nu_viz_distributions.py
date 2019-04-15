import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

if not os.path.exists('./nu_charts'):
    os.mkdir('./nu_charts')

scale = (10, 10)

for mod in modules:

    base_df = pd.read_csv(f'./nu_results/{mod}_base_results.csv')
    kmeans_df = pd.read_csv(f'./nu_results/{mod}_kmeans_results.csv')
    pca_df = pd.read_csv(f'./nu_results/{mod}_pca_results.csv')
    smote_df = pd.read_csv(f'./nu_results/{mod}_smote_results.csv')

    df = base_df.append(kmeans_df).append(pca_df).append(smote_df)
    df['bucket'] = round(df['score'], 1)

    acc_df = df[df['metric'] == 'accuracy']
    fscore_df = df[df['metric'] == 'fscore']
    prec_df = df[df['metric'] == 'precision']
    rec_df = df[df['metric'] == 'recall']

    for met_df in [acc_df, fscore_df, prec_df, rec_df]:

        metric = met_df['metric'].unique()[0].capitalize()

        leq_1 = len(met_df[(met_df['score'] >= 0.0) & (met_df['score'] <= 0.1)].index)
        leq_2 = len(met_df[(met_df['score'] > 0.1) & (met_df['score'] <= 0.2)].index)
        leq_3 = len(met_df[(met_df['score'] > 0.2) & (met_df['score'] <= 0.3)].index)
        leq_4 = len(met_df[(met_df['score'] > 0.3) & (met_df['score'] <= 0.4)].index)
        leq_5 = len(met_df[(met_df['score'] > 0.4) & (met_df['score'] <= 0.5)].index)
        leq_6 = len(met_df[(met_df['score'] > 0.5) & (met_df['score'] <= 0.6)].index)
        leq_7 = len(met_df[(met_df['score'] > 0.6) & (met_df['score'] <= 0.7)].index)
        leq_8 = len(met_df[(met_df['score'] > 0.7) & (met_df['score'] <= 0.8)].index)
        leq_9 = len(met_df[(met_df['score'] > 0.8) & (met_df['score'] <= 0.9)].index)
        leq_10 = len(met_df[(met_df['score'] > 0.9) & (met_df['score'] <= 1.0)].index)

        data = [['0.0 - 0.1', leq_1],
                ['0.1 - 0.2', leq_2],
                ['0.2 - 0.3', leq_3],
                ['0.3 - 0.4', leq_4],
                ['0.4 - 0.5', leq_5],
                ['0.5 - 0.6', leq_6],
                ['0.6 - 0.7', leq_7],
                ['0.7 - 0.8', leq_8],
                ['0.8 - 0.9', leq_9],
                ['0.9 - 1.0', leq_10]]

        counts = pd.DataFrame(columns=['bucket', 'count'], data=data)

        sns.set(style='darkgrid')
        _, ax = plt.subplots(figsize=scale)

        if metric == 'Accuracy':
            color = '#4682b4'
        elif metric == 'Fscore':
            color = '#ff8c00'
        elif metric == 'Precision':
            color = '#90ee90'
        elif metric == 'Recall':
            color = '#b22222'

        mod_plot = sns.barplot(ax=ax, x='bucket', y='count', color=color, data=counts)
        mod_plot.set_title(f"{mod.upper()} : {metric}", fontsize=50)

        plt.ylabel('')
        plt.xlabel('')

        mod_plot.figure.savefig(f'./nu_charts/dist_{mod}_{metric.lower()}.png')

        plt.clf()
        plt.close('all')

        print('FINISHED [ {0} ] : ( {1} )'.format(mod.upper(), metric))
