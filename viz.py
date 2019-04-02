import pandas as pd
import seaborn as sns

dt_df = pd.read_csv('./results/dt-results.csv')
knn_df = pd.read_csv('./results/knn-results.csv')
nb_df = pd.read_csv('./results/nb-results.csv')
rf_df = pd.read_csv('./results/rf-results.csv')
svm_df = pd.read_csv('./results/svm-results.csv')

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

    # if mod == '':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(f'\n+-----+\n| {mod.upper()} |\n+-----+\n')

        # print(acc_df.sort_values(by='score'), '\n')
        # print(fscore_df.sort_values(by='score'), '\n')
        # print(prec_df.sort_values(by='score'), '\n')
        print(rec_df.sort_values(by='score'), '\n')