import os
import time

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix
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

samples = ['non-smote', 'smote']

attributes = {'asmt': assessment_attributes,
              'asmt_stdnt': assessment_attributes + student_attributes,
              'asmt_abd': assessment_attributes + activity_attributes_by_days,
              'asmt_abi': assessment_attributes + activity_attributes_by_interval,
              'asmt_stdnt_abd': assessment_attributes + student_attributes + activity_attributes_by_days,
              'asmt_stdnt_abi': assessment_attributes + student_attributes + activity_attributes_by_interval}

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']

with open('./dt-results.csv', 'w') as file:
    file.write('module,attributes,sample,accuracy,fscore,precision,recall\n')

for smpl in samples:

    for mod in modules:

        # Record start time for the module
        mod_start = time.time()

        # Read the appropriate CSV file into a dataframe
        mod_frame = pd.read_csv(f'./data/_composite/{mod}_composite.csv')

        # Drop rows containing one or more missing value
        mod_frame = mod_frame.dropna()

        # Drop columns irrelevant to classification
        mod_frame = mod_frame.drop(columns=['code_module', 'id_student', 'id_assessment'])

        # Fix typo in original data set for consistency
        mod_frame['imd_band'] = mod_frame['imd_band'].transform(lambda x: '10-20%' if x == '10-20' else x)

        # Fix columns that were cast as float64 to int64
        mod_frame['date'] = mod_frame['date'].astype(np.int64)
        mod_frame['due_vs_submission_date'] = mod_frame['due_vs_submission_date'].astype(np.int64)

        # Recast imd_band as an ordinal attribute
        mod_frame['imd_band'] = mod_frame['imd_band'].transform(lambda x: int(x.split('-')[1].split('%')[0]) / 100)

        # Recast age_band as an ordinal attribute
        mod_frame['age_band'] = mod_frame['age_band'].transform(lambda x: 0.0 if x == '0-35' else (0.5 if x == '35-55' else 1.0))

        # Normalize non-object and non-score columns
        for column in list(mod_frame):
            if mod_frame[column].dtypes != np.object and column != 'score':
                if mod_frame[column].max() == mod_frame[column].min():
                    mod_frame[column] = 0
                else:
                    mod_frame[column] = (mod_frame[column] - mod_frame[column].mean()) / (mod_frame[column].max() - mod_frame[column].std())

        for atbt in attributes:

            # Record start time for the attribute subset
            atbt_start = time.time()

            # Retain only the selected subset of attributes
            atbt_frame = mod_frame[['code_presentation'] + attributes[atbt] + ['score']]

            # Use one-hot encoding for the categorical attributes (other than code_presentation)
            enc_cols = [x for x in list(atbt_frame) if x != 'code_presentation' and atbt_frame[x].dtype == np.object]
            atbt_frame = pd.get_dummies(atbt_frame, columns=enc_cols)

            # Set aside the most recent presentation (2014J) as the testing set
            test = atbt_frame[atbt_frame['code_presentation'] == '2014J']

            # Retain data from all previous semesters as the training set
            train = atbt_frame[atbt_frame['code_presentation'] != '2014J']

            # Drop columns irrelevant to classification
            test = test.drop(columns=['code_presentation'])
            train = train.drop(columns=['code_presentation'])

            # Separate predictive attributes from classification labels
            # Cast score attribute as binary 'pass' / 'fail'
            X_train = train.drop(columns=['score'])
            y_train = train['score'].apply(lambda x: 'pass' if x >= 40 else 'fail')

            X_test = test.drop(columns=['score'])
            y_test = test['score'].apply(lambda x: 'pass' if x >= 40 else 'fail')

            if smpl == 'smote':
                X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)

            # Initialize and train the Decision Tree classifier
            dt = DecisionTreeClassifier(random_state=0)
            dt.fit(X_train, y_train)

            # Predict classes for the test set
            y_hat = dt.predict(X_test)

            # Generate the confusion matrix for the predictions
            tn, fp, fn, tp = confusion_matrix(y_hat, y_test).ravel()

            # Calculate performance metrics from the confusion matrix
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = (tp) / (tp + fp)
            recall = (tp) / (tp + fn)
            fscore = (2 * tp) / ((2 * tp) + tn + fp + fn)

            atbt_end = time.time()

            print(f'\n{mod.upper()} : {atbt.upper()} ({smpl.upper()})')
            print('--------------------')
            print(f'ACCURACY:  \t{round(accuracy, 2)}')
            print(f'F-SCORE:   \t{round(fscore, 2)}')
            print(f'PRECISION: \t{round(precision, 2)}')
            print(f'RECALL:    \t{round(recall, 2)}')
            print('--------------------')
            print(f'[{round(atbt_end - atbt_start, 2)} sec]')

            with open('./dt-results.csv', 'a') as file:
                file.write(f'{mod},{atbt},{smpl},{accuracy},{fscore},{precision},{recall}\n')

        mod_end = time.time()
            