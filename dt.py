import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix as cmat
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import export_graphviz as eg
from sklearn.tree import DecisionTreeClassifier

student_attributes = ['gender',
                      'region',
                      'highest_education',
                      'imd_band',
                      'age_band',
                      'num_of_prev_attempts',
                      'studied_credits',
                      'disability']

assessment_attributes = ['assessment_type',
                         'date',
                         'weight']

activity_attributes_by_days = ['due_vs_submission_date',
                               'resource_clicks_by_days',
                               'resource_clicks_by_days_change',
                               'oucontent_clicks_by_days',
                               'oucontent_clicks_by_days_change',
                               'url_clicks_by_days',
                               'url_clicks_by_days_change',
                               'homepage_clicks_by_days',
                               'homepage_clicks_by_days_change',
                               'subpage_clicks_by_days',
                               'subpage_clicks_by_days_change',
                               'glossary_clicks_by_days',
                               'glossary_clicks_by_days_change',
                               'forumng_clicks_by_days',
                               'forumng_clicks_by_days_change',
                               'oucollaborate_clicks_by_days',
                               'oucollaborate_clicks_by_days_change',
                               'dataplus_clicks_by_days',
                               'dataplus_clicks_by_days_change',
                               'quiz_clicks_by_days',
                               'quiz_clicks_by_days_change',
                               'ouelluminate_clicks_by_days',
                               'ouelluminate_clicks_by_days_change',
                               'sharedsubpage_clicks_by_days',
                               'sharedsubpage_clicks_by_days_change',
                               'questionnaire_clicks_by_days',
                               'questionnaire_clicks_by_days_change',
                               'page_clicks_by_days',
                               'page_clicks_by_days_change',
                               'externalquiz_clicks_by_days',
                               'externalquiz_clicks_by_days_change',
                               'ouwiki_clicks_by_days',
                               'ouwiki_clicks_by_days_change',
                               'dualpane_clicks_by_days',
                               'dualpane_clicks_by_days_change',
                               'repeatactivity_clicks_by_days',
                               'repeatactivity_clicks_by_days_change',
                               'folder_clicks_by_days',
                               'folder_clicks_by_days_change',
                               'htmlactivity_clicks_by_days',
                               'htmlactivity_clicks_by_days_change']

activity_attributes_by_interval = ['due_vs_submission_date',
                                   'resource_clicks_by_interval',
                                   'resource_clicks_by_interval_change',
                                   'oucontent_clicks_by_interval',
                                   'oucontent_clicks_by_interval_change',
                                   'url_clicks_by_interval',
                                   'url_clicks_by_interval_change',
                                   'homepage_clicks_by_interval',
                                   'homepage_clicks_by_interval_change',
                                   'subpage_clicks_by_interval',
                                   'subpage_clicks_by_interval_change',
                                   'glossary_clicks_by_interval',
                                   'glossary_clicks_by_interval_change',
                                   'forumng_clicks_by_interval',
                                   'forumng_clicks_by_interval_change',
                                   'oucollaborate_clicks_by_interval',
                                   'oucollaborate_clicks_by_interval_change',
                                   'dataplus_clicks_by_interval',
                                   'dataplus_clicks_by_interval_change',
                                   'quiz_clicks_by_interval',
                                   'quiz_clicks_by_interval_change',
                                   'ouelluminate_clicks_by_interval',
                                   'ouelluminate_clicks_by_interval_change',
                                   'sharedsubpage_clicks_by_interval',
                                   'sharedsubpage_clicks_by_interval_change',
                                   'questionnaire_clicks_by_interval',
                                   'questionnaire_clicks_by_interval_change',
                                   'page_clicks_by_interval',
                                   'page_clicks_by_interval_change',
                                   'externalquiz_clicks_by_interval',
                                   'externalquiz_clicks_by_interval_change',
                                   'ouwiki_clicks_by_interval',
                                   'ouwiki_clicks_by_interval_change',
                                   'dualpane_clicks_by_interval',
                                   'dualpane_clicks_by_interval_change',
                                   'repeatactivity_clicks_by_interval',
                                   'repeatactivity_clicks_by_interval_change',
                                   'folder_clicks_by_interval',
                                   'folder_clicks_by_interval_change',
                                   'htmlactivity_clicks_by_interval',
                                   'htmlactivity_clicks_by_interval_change']

# if os.path.exists('./dt-output'):
#     shutil.rmtree('./dt-output')

# if os.path.exists('./trees'):
#     shutil.rmtree('./trees')

# os.mkdir('./dt-output')
# os.mkdir('./dt-output/non-smote')
# os.mkdir('./dt-output/smote')

# os.mkdir('./trees')
# os.mkdir('./trees/non-smote')
# os.mkdir('./trees/smote')

sample = 'non-smote'
# sample = 'smote'

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']

titles = ['asmt', 'asmt_stdnt', 'asmt_abd', 'asmt_abi', 'asmt_stdnt_abd', 'asmt_stdnt_abi']

for title in titles:

    for mod in modules:

        with open(f'./dt-output/{sample}/{mod}-{title}.txt', 'w') as file:
            file.write(f'{mod}-{title}\n\n')

        for depth in range(1,11):

            mod_start = time.time()

            print(f'\nBeginning Module {mod.upper()} ({title}, depth {depth}) ...')

            # Read the appropriate CSV file as a dataframe
            mod_frame = pd.read_csv(f'./data/_composite/{mod}_composite.csv')

            # Separate the class column and convert scores to 'pass' or 'fail'
            y = mod_frame['score'].apply(lambda x: 'pass' if x >= 40 else 'fail')

            # Drop columns that are irrelevant to classification
            drop_cols = ['code_module',
                        'code_presentation',
                        'id_student',
                        'id_assessment',
                        'score']

            X = mod_frame.drop(columns=drop_cols)

            # Fix typo in original data for consistency
            X['imd_band'] = X['imd_band'].transform(
                lambda x: '10-20%' if x == '10-20' else x)

            if title == 'asmt':
                X = X[assessment_attributes]
            elif title == 'asmt_stdnt':
                X = X[assessment_attributes + student_attributes]
            elif title == 'asmt_abd':
                X = X[assessment_attributes + activity_attributes_by_days]
            elif title == 'asmt_abi':
                X = X[assessment_attributes + activity_attributes_by_interval]
            elif title == 'asmt_stdnt_abd':
                X = X[assessment_attributes + student_attributes + activity_attributes_by_days]
            elif title == 'asmt_stndt_abi':
                X = X[assessment_attributes + student_attributes + activity_attributes_by_interval]

            # Use one-hot-encoding for categorical variables
            X = pd.get_dummies(X)

            # NORMALIZATION

            # Initialize for stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

            aggregate_accuracy = 0
            aggregate_precision = 0
            aggregate_recall = 0
            aggregate_fscore = 0

            fold = 0

            for train_index, test_index in skf.split(X, y):

                fold += 1

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                X_smote, y_smote = SMOTE(random_state=0).fit_resample(X_train, y_train)

                dt = DecisionTreeClassifier(max_depth=depth, random_state=0)

                # norm_pass = 0
                # norm_fail = 0

                # smote_pass = 0
                # smote_fail = 0

                # for item in y_train:
                #     if item == 'pass':
                #         norm_pass += 1
                #     else:
                #         norm_fail += 1

                # for item in y_smote:
                #     if item == 'pass':
                #         smote_pass += 1
                #     else:
                #         smote_fail += 1

                # print(f'NORM:  {norm_pass} pass, {norm_fail} fail ({round(((norm_pass) / (norm_pass + norm_fail)) * 100, 2)}% pass)')
                # print(f'SMOTE: {smote_pass} pass, {smote_fail} fail ({round(((smote_pass) / (smote_pass + smote_fail)) * 100, 2)}% pass)')
                # print('----------')

                if sample == 'non-smote':
                    dt.fit(X_train, y_train)
                elif sample == 'smote':
                    dt.fit(X_smote, y_smote)

                eg(dt, out_file=f'./trees/{sample}/{mod}-{title}-{depth}.dot', feature_names=list(X_train), filled=True)
                os.system(f'dot -Tpng ./trees/{sample}/{mod}-{title}-{depth}.dot -o ./trees/{sample}/_{mod}-{title}-{depth}.png')

                y_hat = dt.predict(X_test)

                tn, fp, fn, tp = cmat(y_hat, y_test).ravel()

                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = (tp) / (tp + fp)
                recall = (tp) / (tp + fn)
                fscore = (2 * tp) / ((2 * tp) + tn + fp + fn)

                aggregate_accuracy += accuracy
                aggregate_precision += precision
                aggregate_recall += recall
                aggregate_fscore += fscore

                # print(f'\n{mod.upper()}-{fold}')
                # print('--------------------')
                # print(f'ACCURACY:  \t{round(accuracy, 2)}')
                # print(f'PRECISION: \t{round(precision, 2)}')
                # print(f'RECALL:    \t{round(recall, 2)}')
                # print(f'F-SCORE:   \t{round(fscore, 2)}')
                # print('--------------------')

                print(f'\t... Finished module {mod.upper()}, fold {fold}')

            aggregate_accuracy = aggregate_accuracy / 10
            aggregate_precision = aggregate_precision / 10
            aggregate_recall = aggregate_recall / 10
            aggregate_fscore = aggregate_fscore / 10

            with open(f'./dt-output/{sample}/{mod}-{title}.txt', 'a') as file:
                file.write(f'[{depth}] Accuracy:  {aggregate_accuracy}\n')
                file.write(f'[{depth}] Precision: {aggregate_precision}\n')
                file.write(f'[{depth}] Recall:    {aggregate_recall}\n')
                file.write(f'[{depth}] F-Score:   {aggregate_fscore}\n\n')

            mod_end = time.time()

            print(f'Module {mod.upper()} ({title}, depth {depth}) DONE [{round(mod_end - mod_start)} sec]\n\n')
