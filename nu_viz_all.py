import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

classifiers = ['dt', 'knn', 'nb', 'nn', 'rf', 'svm']

if not os.path.exists('./nu_charts'):
    os.mkdir('./nu_charts')

scale = (6.25, 5)

for mod in modules:

    base_df = pd.read_csv(f'./nu_results/{mod}_base_results.csv')
    kmeans_df = pd.read_csv(f'./nu_results/{mod}_kmeans_results.csv')
    pca_df = pd.read_csv(f'./nu_results/{mod}_pca_results.csv')
    smote_df = pd.read_csv(f'./nu_results/{mod}_smote_results.csv')

    df = base_df.append(kmeans_df).append(pca_df).append(smote_df)

    df['id'] = df['classifier'] + ' (' + df['attributes'] + ')'

    for clf in classifiers:

        clf_df = df[df['classifier'] == clf]

        # clf_df = clf_df.drop(columns=['module', 'attributes', 'classifier'])

        key = 'id'

        acc_df = clf_df[clf_df['metric'] == 'accuracy']
        acc_df = acc_df.sort_values(by=key)

        fscore_df = clf_df[clf_df['metric'] == 'fscore']
        fscore_df = fscore_df.sort_values(by=key)

        prec_df = clf_df[clf_df['metric'] == 'precision']
        prec_df = prec_df.sort_values(by=key)

        rec_df = clf_df[clf_df['metric'] == 'recall']
        rec_df = rec_df.sort_values(by=key)

        for met_df in [acc_df, fscore_df, prec_df, rec_df]:

            metric = met_df['metric'].unique()[0].capitalize()
            met_df = met_df.drop(columns=['metric'])

            sns.set(style='darkgrid')
            _, ax = plt.subplots(figsize=scale)

            colors = {'base': '#4682b4',
                      'kmeans': '#ff8c00',
                      'pca': '#90ee90',
                      'smote': '#b22222'}

            mod_plot = sns.barplot(ax=ax, x='attributes', y='score', hue='technique', hue_order=[
                                   'base', 'kmeans', 'pca', 'smote'], palette=colors, data=met_df)
            mod_plot.set_title(
                f"{mod.upper()} : {clf.upper()} : {metric}", fontsize=25)

            mod_plot.set(ylim=(0,1))

            ax.get_legend().remove()

            plt.xticks(rotation=90)
            plt.ylabel('Metric Score')
            plt.xlabel('Model')
            plt.tick_params(labelsize=10)
            
            # plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

            mod_plot.figure.savefig(
                f'./nu_charts/clf_{mod}_{clf}_{metric.lower()}.png')

            plt.clf()
            plt.close('all')

            print(
                'FINISHED [ {0}-{1} ] : ( {2} )'.format(mod.upper(), clf.upper(), metric))
