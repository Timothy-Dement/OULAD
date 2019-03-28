import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

scale = (6.5, 4)

options = ['compare_attributes', 'compare_samples']

if not os.path.exists('./charts'):
    os.mkdir('./charts')

results_file = './dt-results.csv'

df = pd.read_csv(results_file)

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']

for option in options:

    if option == 'compare_attributes':

        smpl_df = df[df['sample'] == 'non-smote']

        for mod in modules:

            mod_df = smpl_df[smpl_df['module'] == mod]

            accs = mod_df[mod_df['metric'] == 'accuracy']
            fscores = mod_df[mod_df['metric'] == 'fscore']
            precs = mod_df[mod_df['metric'] == 'precision']
            recs = mod_df[mod_df['metric'] == 'recall']

            min_acc, max_acc = accs['score'].min(), accs['score'].max()
            min_fscore, max_fscore = fscores['score'].min(), fscores['score'].max()
            min_prec, max_prec = precs['score'].min(), precs['score'].max()
            min_rec, max_rec = recs['score'].min(), recs['score'].max()

            norm_df = mod_df.copy(deep=True)

            for index, row in norm_df.iterrows():
                if row.loc['metric'] == 'accuracy':
                    norm_df.at[index, 'score'] = (row.loc['score'] - min_acc) / (max_acc - min_acc)
                elif row.loc['metric'] == 'fscore':
                    norm_df.at[index, 'score'] = (row.loc['score'] - min_fscore) / (max_fscore - min_fscore)
                elif row.loc['metric'] == 'precision':
                    norm_df.at[index, 'score'] = (row.loc['score'] - min_prec) / (max_prec - min_prec)
                elif row.loc['metric'] == 'recall':
                    norm_df.at[index, 'score'] = (row.loc['score'] - min_rec) / (max_rec - min_rec)

            acc_win = accs.loc[accs['score'].idxmax()]
            fscore_win = fscores.loc[fscores['score'].idxmax()]
            prec_win = precs.loc[precs['score'].idxmax()]
            rec_win = recs.loc[recs['score'].idxmax()]

            print(f'{mod.upper()} : NON-SMOTE')
            print('--------------------')
            print(f'ACCURACY:  \t{round(acc_win["score"], 2)} \t{acc_win["attributes"]}')
            print(f'F-SCORE:   \t{round(fscore_win["score"], 2)} \t{fscore_win["attributes"]}')
            print(f'PRECISION: \t{round(prec_win["score"], 2)} \t{prec_win["attributes"]}')
            print(f'RECALL:    \t{round(rec_win["score"], 2)} \t{rec_win["attributes"]}')
            print('--------------------\n')

            sns.set(style='whitegrid')
            _, ax = plt.subplots(figsize=scale)

            mod_plot = sns.barplot(ax=ax, x='score', y='attributes', hue='metric', data=norm_df)

            if mod == 'zzz':
                mod_plot.set_title(f'All Modules Combined')
            else:
                mod_plot.set_title(f'Module {mod.upper()}')

            plt.xticks(rotation=90)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.ylabel('Attribute Subset')
            plt.xlabel('Normalized Metric Score')
            plt.tight_layout()

            y = plt.gca().axes.get_ylim()
            plt.plot((0,0), y, linewidth='2', color='black')


            mod_plot.figure.savefig(f'./charts/atbt_{mod}.png')

    elif option == 'compare_samples':

        for mod in modules:

            mod_df = df[df['module'] == mod]

            nonsmote_df = mod_df[mod_df['sample'] == 'non-smote'].reset_index()
            smote_df = mod_df[mod_df['sample'] == 'smote'].reset_index()

            accs = smote_df[smote_df['metric'] == 'accuracy']
            fscores = smote_df[smote_df['metric'] == 'fscore']
            precs = smote_df[smote_df['metric'] == 'precision']
            recs = smote_df[smote_df['metric'] == 'recall']

            acc_win = accs.loc[accs['score'].idxmax()]
            fscore_win = fscores.loc[fscores['score'].idxmax()]
            prec_win = precs.loc[precs['score'].idxmax()]
            rec_win = recs.loc[recs['score'].idxmax()]

            print(f'{mod.upper()} : SMOTE')
            print('--------------------')
            print(f'ACCURACY:  \t{round(acc_win["score"], 2)} \t{acc_win["attributes"]}')
            print(f'F-SCORE:   \t{round(fscore_win["score"], 2)} \t{fscore_win["attributes"]}')
            print(f'PRECISION: \t{round(prec_win["score"], 2)} \t{prec_win["attributes"]}')
            print(f'RECALL:    \t{round(rec_win["score"], 2)} \t{rec_win["attributes"]}')
            print('--------------------\n')

            diff_df = smote_df.copy(deep=True)

            for index, row in diff_df.iterrows():
                diff_df.at[index, 'score'] = smote_df.at[index, 'score'] - nonsmote_df.at[index, 'score']

            sns.set(style='whitegrid')
            _, ax = plt.subplots(figsize=scale)

            mod_plot = sns.barplot(ax=ax, x='score', y='attributes', hue='metric', data=diff_df)

            if mod == 'zzz':
                mod_plot.set_title(f'All Modules Combined')
            else:
                mod_plot.set_title(f'Module {mod.upper()}')

            plt.xticks(rotation=90)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.ylabel('Attribute Subset')
            plt.xlabel('Metric Change with SMOTE')
            plt.tight_layout()

            y = plt.gca().axes.get_ylim()
            plt.plot((0,0), y, linewidth='2', color='black')

            mod_plot.figure.savefig(f'./charts/smote_{mod}.png')
