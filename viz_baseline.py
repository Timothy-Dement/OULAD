import os

from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

import pandas as pd
import seaborn as sns

dt_df = pd.read_csv('./results/dt-results.csv')
knn_df = pd.read_csv('./results/knn-results.csv')
nb_df = pd.read_csv('./results/nb-results.csv')
rf_df = pd.read_csv('./results/rf-results.csv')
svm_df = pd.read_csv('./results/svm-results.csv')

if not os.path.exists('./charts'):
    os.mkdir('./charts')

scale = (20, 5)

for mod in ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']:

    mod_dt = dt_df[dt_df['module'] == mod].copy(deep=True)
    mod_knn = knn_df[knn_df['module'] == mod].copy(deep=True)
    mod_nb = nb_df[nb_df['module'] == mod].copy(deep=True)
    mod_rf = rf_df[rf_df['module'] == mod].copy(deep=True)
    mod_svm = svm_df[svm_df['module'] == mod].copy(deep=True)

    mod_dt['id'] = mod_dt['sample'] + '-dt-' + mod_dt['attributes']
    mod_knn['id'] = mod_knn['sample'] + '-knn-' + mod_knn['attributes']
    mod_nb['id'] = mod_nb['sample'] + '-nb-' + mod_nb['attributes']
    mod_rf['id'] = mod_rf['sample'] + '-rf-' + mod_rf['attributes']
    mod_svm['id'] = mod_svm['sample'] + '-svm-' + mod_svm['attributes']

    mod_dt = mod_dt.drop(columns=['module', 'sample', 'attributes'])
    mod_knn = mod_knn.drop(columns=['module', 'sample', 'attributes'])
    mod_nb = mod_nb.drop(columns=['module', 'sample', 'attributes'])
    mod_rf = mod_rf.drop(columns=['module', 'sample', 'attributes'])
    mod_svm = mod_svm.drop(columns=['module', 'sample', 'attributes'])

    master = pd.DataFrame(columns=list(mod_dt))

    master = master.append(mod_dt, ignore_index=True)
    master = master.append(mod_knn, ignore_index=True)
    master = master.append(mod_nb, ignore_index=True)
    master = master.append(mod_rf, ignore_index=True)
    master = master.append(mod_svm, ignore_index=True)

    acc_df = master[master['metric'] == 'accuracy']
    fscore_df = master[master['metric'] == 'fscore']
    prec_df = master[master['metric'] == 'precision']
    rec_df = master[master['metric'] == 'recall']

    acc_df = acc_df.sort_values(by='score', ascending=False)
    fscore_df = fscore_df.sort_values(by='score', ascending=False)
    prec_df = prec_df.sort_values(by='score', ascending=False)
    rec_df = rec_df.sort_values(by='score', ascending=False)

    for metric_df in [acc_df, fscore_df, prec_df, rec_df]:

        metric = metric_df['metric'].unique()[0].capitalize()

        sns.set(style='whitegrid')
        _, ax = plt.subplots(figsize=scale)

        mod_plot = sns.barplot(ax=ax, x='id', y='score', data=metric_df, color='#008080')

        if mod == 'zzz':
            mod_plot.set_title(f"{metric} - All Modules Combined", fontsize=15)
        else:
            mod_plot.set_title(f"{metric} - Module {mod.upper()}", fontsize=15)

        plt.xticks(rotation=90)
        plt.ylabel('Metric Score')
        plt.xlabel('Model')
        plt.tick_params(labelsize=8)

        mod_plot.figure.savefig(f'./charts/bl-{mod}-{metric.lower()}.png')

        plt.clf()
        plt.close('all')
