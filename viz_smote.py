import os

from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

import pandas as pd
import seaborn as sns

dt_base = pd.read_csv('./results/dt_results.csv')
knn_base = pd.read_csv('./results/knn_results.csv')
nb_base = pd.read_csv('./results/nb_results.csv')
rf_base = pd.read_csv('./results/rf_results.csv')
svm_base = pd.read_csv('./results/svm_results.csv')

dt_smote = pd.read_csv('./results/dt_smote_results.csv')
knn_smote = pd.read_csv('./results/knn_smote_results.csv')
nb_smote = pd.read_csv('./results/nb_smote_results.csv')
rf_smote = pd.read_csv('./results/rf_smote_results.csv')
svm_smote = pd.read_csv('./results/svm_smote_results.csv')

for df in [dt_base, knn_base, nb_base, rf_base, svm_base]:
    df['sample'] = 'none'

dt_df = dt_base.append(dt_smote)
knn_df = knn_base.append(knn_smote)
nb_df = nb_base.append(nb_smote)
rf_df = rf_base.append(rf_smote)
svm_df = svm_base.append(svm_smote)

if not os.path.exists('./charts'):
    os.mkdir('./charts')

scale = (20, 5)

for mod in ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']:

    mod_dt = dt_df[dt_df['module'] == mod].copy(deep=True)
    mod_knn = knn_df[knn_df['module'] == mod].copy(deep=True)
    mod_nb = nb_df[nb_df['module'] == mod].copy(deep=True)
    mod_rf = rf_df[rf_df['module'] == mod].copy(deep=True)
    mod_svm = svm_df[svm_df['module'] == mod].copy(deep=True)

    mod_dt['id'] = 'dt-' + mod_dt['attributes']
    mod_knn['id'] = 'knn-' + mod_knn['attributes']
    mod_nb['id'] = 'nb-' + mod_nb['attributes']
    mod_rf['id'] = 'rf-' + mod_rf['attributes']
    mod_svm['id'] = 'svm-' + mod_svm['attributes']

    mod_dt = mod_dt.drop(columns=['module', 'attributes'])
    mod_knn = mod_knn.drop(columns=['module', 'attributes'])
    mod_nb = mod_nb.drop(columns=['module', 'attributes'])
    mod_rf = mod_rf.drop(columns=['module', 'attributes'])
    mod_svm = mod_svm.drop(columns=['module', 'attributes'])

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

    acc_df = acc_df.sort_values(by='id')
    fscore_df = fscore_df.sort_values(by='id')
    prec_df = prec_df.sort_values(by='id')
    rec_df = rec_df.sort_values(by='id')

    for metric_df in [acc_df, fscore_df, prec_df, rec_df]:

        metric = metric_df['metric'].unique()[0].capitalize()

        sns.set(style='whitegrid')
        _, ax = plt.subplots(figsize=scale)

        mod_plot = sns.barplot(ax=ax, x='id', y='score', hue='sample', hue_order=['none', 'smote'], data=metric_df)

        if mod == 'zzz':
            mod_plot.set_title(f"{metric} - All Modules Combined", fontsize=15)
        else:
            mod_plot.set_title(f"{metric} - Module {mod.upper()}", fontsize=15)

        plt.xticks(rotation=90)
        plt.ylabel('Metric Score')
        plt.xlabel('Model')
        plt.tick_params(labelsize=10)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        mod_plot.figure.savefig(f'./charts/smote-{mod}-{metric.lower()}.png')

        plt.clf()
        plt.close('all')
