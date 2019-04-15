import pandas as pd

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    for mod in modules:

        base_df = pd.read_csv(f'./nu_results/{mod}_base_results.csv')
        kmeans_df = pd.read_csv(f'./nu_results/{mod}_kmeans_results.csv')
        pca_df = pd.read_csv(f'./nu_results/{mod}_pca_results.csv')
        smote_df = pd.read_csv(f'./nu_results/{mod}_smote_results.csv')

        df = base_df.append(kmeans_df).append(pca_df).append(smote_df)

        acc_df = df[df['metric'] == 'accuracy']
        fscore_df = df[df['metric'] == 'fscore']
        prec_df = df[df['metric'] == 'precision']
        rec_df = df[df['metric'] == 'recall']

        acc_df = acc_df.rename(columns={'score':'accuracy'})
        fscore_df = fscore_df.rename(columns={'score':'fscore'})
        prec_df = prec_df.rename(columns={'score':'precision'})
        rec_df = rec_df.rename(columns={'score':'recall'})

        acc_df = acc_df.drop(columns=['metric'])
        fscore_df = fscore_df.drop(columns=['metric'])
        prec_df = prec_df.drop(columns=['metric'])
        rec_df = rec_df.drop(columns=['metric'])

        master = acc_df.merge(fscore_df, on=[x for x in list(acc_df) if x != 'accuracy'])
        master = master.merge(prec_df, on=[x for x in list(acc_df) if (x != 'accuracy') and (x != 'fscore')])
        master = master.merge(rec_df, on=[x for x in list(acc_df) if (x != 'accuracy') and (x != 'fscore') and (x != 'precision')])

        acc_th = (master['accuracy'].sort_values(ascending=False)).unique()[0:5]
        fscore_th = (master['fscore'].sort_values(ascending=False)).unique()[0:5]
        prec_th = (master['precision'].sort_values(ascending=False)).unique()[0:5]
        rec_th = (master['recall'].sort_values(ascending=False)).unique()[0:5]

        acc_top = master[master['accuracy'] >= acc_th[4]]
        fscore_top = master[master['fscore'] >= fscore_th[4]]
        prec_top = master[master['precision'] >= prec_th[4]]
        rec_top = master[master['recall'] >= rec_th[4]]

        #
        #
        #

        print(f'\n===== {mod.upper()} : ACC ==================================\n')

        best_acc = acc_top[acc_top['accuracy'] == acc_th[0]]

        print('{0:.4f}'.format(acc_th[0]))

        best_acc_tot = len(best_acc.index)

        print(f'\nATTRIBUTES [ {best_acc_tot} ]\n')

        best_acc_atbt_labels = list(best_acc['attributes'].value_counts().index)
        best_acc_atbt_counts = list(best_acc['attributes'].value_counts())

        for i, v in enumerate(best_acc_atbt_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_acc_tot * 100, v, best_acc_atbt_labels[i]))

        print(f'\nCLASSIFIERS [ {best_acc_tot} ]\n')

        best_acc_clf_labels = list(best_acc['classifier'].value_counts().index)
        best_acc_clf_counts = list(best_acc['classifier'].value_counts())

        for i, v in enumerate(best_acc_clf_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_acc_tot * 100, v, best_acc_clf_labels[i]))

        print(f'\nTECHNIQUES [ {best_acc_tot} ]\n')

        best_acc_tech_labels = list(best_acc['technique'].value_counts().index)
        best_acc_tech_counts = list(best_acc['technique'].value_counts())

        for i, v in enumerate(best_acc_tech_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_acc_tot * 100, v, best_acc_tech_labels[i]))
        
        # print(f'\n==================================================\n')

        # for score in acc_th:
        #     print('{0:.4f}'.format(score))

        # acc_tot = len(acc_top.index)

        # print(f'\nATTRIBUTES [ {len(acc_top.index)} ]\n')

        # acc_atbt_labels = list(acc_top['attributes'].value_counts().index)
        # acc_atbt_counts = list(acc_top['attributes'].value_counts())

        # for i, v in enumerate(acc_atbt_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / acc_tot * 100, v, acc_atbt_labels[i]))

        # print(f'\nCLASSIFIERS [ {len(acc_top.index)} ]\n')

        # acc_clf_labels = list(acc_top['classifier'].value_counts().index)
        # acc_clf_counts = list(acc_top['classifier'].value_counts())

        # for i, v in enumerate(acc_clf_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / acc_tot * 100, v, acc_clf_labels[i]))

        # print(f'\nTECHNIQUES [ {len(acc_top.index)} ]\n')

        # acc_tech_labels = list(acc_top['technique'].value_counts().index)
        # acc_tech_counts = list(acc_top['technique'].value_counts())

        # for i, v in enumerate(acc_tech_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / acc_tot * 100, v, acc_tech_labels[i]))

        #
        #
        #

        print(f'\n===== {mod.upper()} : FSCORE ===============================\n')

        best_fscore = fscore_top[fscore_top['fscore'] == fscore_th[0]]

        print('{0:.4f}'.format(fscore_th[0]))

        best_fscore_tot = len(best_fscore.index)

        print(f'\nATTRIBUTES [ {best_fscore_tot} ]\n')

        best_fscore_atbt_labels = list(best_fscore['attributes'].value_counts().index)
        best_fscore_atbt_counts = list(best_fscore['attributes'].value_counts())

        for i, v in enumerate(best_fscore_atbt_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_fscore_tot * 100, v, best_fscore_atbt_labels[i]))

        print(f'\nCLASSIFIERS [ {best_fscore_tot} ]\n')

        best_fscore_clf_labels = list(best_fscore['classifier'].value_counts().index)
        best_fscore_clf_counts = list(best_fscore['classifier'].value_counts())

        for i, v in enumerate(best_fscore_clf_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_fscore_tot * 100, v, best_fscore_clf_labels[i]))

        print(f'\nTECHNIQUES [ {best_fscore_tot} ]\n')

        best_fscore_tech_labels = list(best_fscore['technique'].value_counts().index)
        best_fscore_tech_counts = list(best_fscore['technique'].value_counts())

        for i, v in enumerate(best_fscore_tech_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_fscore_tot * 100, v, best_fscore_tech_labels[i]))

        # print(f'\n==================================================\n')

        # for score in fscore_th:
        #     print('{0:.4f}'.format(score))

        # fscore_tot = len(fscore_top.index)

        # print(f'\nATTRIBUTES [ {len(fscore_top.index)} ]\n')

        # fscore_atbt_labels = list(fscore_top['attributes'].value_counts().index)
        # fscore_atbt_counts = list(fscore_top['attributes'].value_counts())

        # for i, v in enumerate(fscore_atbt_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / fscore_tot * 100, v, fscore_atbt_labels[i]))

        # print(f'\nCLASSIFIERS [ {len(fscore_top.index)} ]\n')

        # fscore_clf_labels = list(fscore_top['classifier'].value_counts().index)
        # fscore_clf_counts = list(fscore_top['classifier'].value_counts())

        # for i, v in enumerate(fscore_clf_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / fscore_tot * 100, v, fscore_clf_labels[i]))

        # print(f'\nTECHNIQUES [ {len(fscore_top.index)} ]\n')

        # fscore_tech_labels = list(fscore_top['technique'].value_counts().index)
        # fscore_tech_counts = list(fscore_top['technique'].value_counts())

        # for i, v in enumerate(fscore_tech_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / fscore_tot * 100, v, fscore_tech_labels[i]))

        #
        #
        #

        print(f'\n===== {mod.upper()} : PREC =================================\n')

        best_prec = prec_top[prec_top['precision'] == prec_th[0]]

        print('{0:.4f}'.format(prec_th[0]))

        best_prec_tot = len(best_prec.index)

        print(f'\nATTRIBUTES [ {best_prec_tot} ]\n')

        best_prec_atbt_labels = list(best_prec['attributes'].value_counts().index)
        best_prec_atbt_counts = list(best_prec['attributes'].value_counts())

        for i, v in enumerate(best_prec_atbt_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_prec_tot * 100, v, best_prec_atbt_labels[i]))

        print(f'\nCLASSIFIERS [ {best_prec_tot} ]\n')

        best_prec_clf_labels = list(best_prec['classifier'].value_counts().index)
        best_prec_clf_counts = list(best_prec['classifier'].value_counts())

        for i, v in enumerate(best_prec_clf_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_prec_tot * 100, v, best_prec_clf_labels[i]))

        print(f'\nTECHNIQUES [ {best_prec_tot} ]\n')

        best_prec_tech_labels = list(best_prec['technique'].value_counts().index)
        best_prec_tech_counts = list(best_prec['technique'].value_counts())

        for i, v in enumerate(best_prec_tech_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_prec_tot * 100, v, best_prec_tech_labels[i]))

        # print(f'\n==================================================\n')

        # for score in prec_th:
        #     print('{0:.4f}'.format(score))

        # prec_tot = len(prec_top.index)

        # print(f'\nATTRIBUTES [ {len(prec_top.index)} ]\n')

        # prec_atbt_labels = list(prec_top['attributes'].value_counts().index)
        # prec_atbt_counts = list(prec_top['attributes'].value_counts())

        # for i, v in enumerate(prec_atbt_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / prec_tot * 100, v, prec_atbt_labels[i]))

        # print(f'\nCLASSIFIERS [ {len(prec_top.index)} ]\n')

        # prec_clf_labels = list(prec_top['classifier'].value_counts().index)
        # prec_clf_counts = list(prec_top['classifier'].value_counts())

        # for i, v in enumerate(prec_clf_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / prec_tot * 100, v, prec_clf_labels[i]))

        # print(f'\nTECHNIQUES [ {len(prec_top.index)} ]\n')

        # prec_tech_labels = list(prec_top['technique'].value_counts().index)
        # prec_tech_counts = list(prec_top['technique'].value_counts())

        # for i, v in enumerate(prec_tech_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / prec_tot * 100, v, prec_tech_labels[i]))

        #
        #
        #

        print(f'\n===== {mod.upper()} : REC ==================================\n')

        best_rec = rec_top[rec_top['recall'] == rec_th[0]]

        print('{0:.4f}'.format(rec_th[0]))

        best_rec_tot = len(best_rec.index)

        print(f'\nATTRIBUTES [ {best_rec_tot} ]\n')

        best_rec_atbt_labels = list(best_rec['attributes'].value_counts().index)
        best_rec_atbt_counts = list(best_rec['attributes'].value_counts())

        for i, v in enumerate(best_rec_atbt_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_rec_tot * 100, v, best_rec_atbt_labels[i]))

        print(f'\nCLASSIFIERS [ {best_rec_tot} ]\n')

        best_rec_clf_labels = list(best_rec['classifier'].value_counts().index)
        best_rec_clf_counts = list(best_rec['classifier'].value_counts())

        for i, v in enumerate(best_rec_clf_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_rec_tot * 100, v, best_rec_clf_labels[i]))

        print(f'\nTECHNIQUES [ {best_rec_tot} ]\n')

        best_rec_tech_labels = list(best_rec['technique'].value_counts().index)
        best_rec_tech_counts = list(best_rec['technique'].value_counts())

        for i, v in enumerate(best_rec_tech_counts):
            print('{0:.2f} \t {1} \t {2}'.format(v / best_rec_tot * 100, v, best_rec_tech_labels[i]))

        # print(f'\n==================================================\n')

        # for score in rec_th:
        #     print('{0:.4f}'.format(score))

        # rec_tot = len(rec_top.index)

        # print(f'\nATTRIBUTES [ {len(rec_top.index)} ]\n')

        # rec_atbt_labels = list(rec_top['attributes'].value_counts().index)
        # rec_atbt_counts = list(rec_top['attributes'].value_counts())

        # for i, v in enumerate(rec_atbt_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / rec_tot * 100, v, rec_atbt_labels[i]))

        # print(f'\nCLASSIFIERS [ {len(rec_top.index)} ]\n')

        # rec_clf_labels = list(rec_top['classifier'].value_counts().index)
        # rec_clf_counts = list(rec_top['classifier'].value_counts())

        # for i, v in enumerate(rec_clf_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / rec_tot * 100, v, rec_clf_labels[i]))

        # print(f'\nTECHNIQUES [ {len(rec_top.index)} ]\n')

        # rec_tech_labels = list(rec_top['technique'].value_counts().index)
        # rec_tech_counts = list(rec_top['technique'].value_counts())

        # for i, v in enumerate(rec_tech_counts):
        #     print('{0:.2f} \t {1} \t {2}'.format(v / rec_tot * 100, v, rec_tech_labels[i]))

    print('\n==================================================')
