import os
import time

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

all_start = time.time()

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

classifiers = ['dt', 'svm', 'nb', 'knn', 'rf']

attributes = {'asmt': assessment_attributes,
              'stdnt': student_attributes,
              'abd': activity_attributes_by_days,
              'abi': activity_attributes_by_interval,
              'asmt_stdnt': assessment_attributes + student_attributes,
              'asmt_abd': assessment_attributes + activity_attributes_by_days,
              'asmt_abi': assessment_attributes + activity_attributes_by_interval,
              'stdnt_abd': student_attributes + activity_attributes_by_days,
              'stdnt_abi': student_attributes + activity_attributes_by_interval,
              'asmt_stdnt_abd': assessment_attributes + student_attributes + activity_attributes_by_days,
              'asmt_stdnt_abi': assessment_attributes + student_attributes + activity_attributes_by_interval}

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'zzz']

if not os.path.exists('./results'):
    os.mkdir('./results')

for clf in classifiers:

    with open(f'./results/{clf}_results.csv', 'w') as file:
        file.write('module,attributes,sample,metric,score\n')

    with open(f'./results/{clf}_smote_results.csv', 'w') as file:
        file.write('module,attributes,sample,metric,score\n')

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

            def he_transform(value):
                if value == 'Post Graduate Qualification':
                    return 1.00
                elif value == 'HE Qualification':
                    return 0.75
                elif value == 'A Level or Equivalent':
                    return 0.50
                elif value == 'Lower Than A Level':
                    return 0.25
                else:
                    return 0.00

            # Recast highest_education as an ordinal attribute
            mod_frame['highest_education'] = mod_frame['highest_education'].transform(he_transform)

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
                # Cast score attribute as binary 'pass' (0) / 'fail' (1)
                X_train = train.drop(columns=['score'])
                y_train = train['score'].apply(lambda x: 0 if x >= 40 else 1)

                X_test = test.drop(columns=['score'])
                y_test = test['score'].apply(lambda x: 0 if x >= 40 else 1)

                if smpl == 'smote':
                    X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)

                # Initialize and train the Decision Tree classifier
                # dt = DecisionTreeClassifier(random_state=0)
                # dt.fit(X_train, y_train)

                model = None

                if clf == 'dt':
                    model = DecisionTreeClassifier(random_state=0)
                elif clf == 'svm':
                    model = SVC(gamma='auto', random_state=0)
                elif clf == 'nb':
                    model = GaussianNB()
                elif clf == 'knn':
                    model = KNeighborsClassifier(n_jobs=-1)
                elif clf == 'rf':
                    model = RandomForestClassifier(n_jobs=-1, random_state='0')

                model.fit(X_train, y_train)

                # Predict classes for the test set
                y_hat = model.predict(X_test)

                # Generate the confusion matrix for the predictions
                tn, fp, fn, tp = confusion_matrix(y_hat, y_test).ravel()

                # Calculate performance metrics from the confusion matrix

                accuracy = None
                precision = None
                recall = None
                fscore = None

                if (tp + tn + fp + fn) == 0:
                    accuracy = 0.0
                else:
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                if (tp + fp) == 0:
                    precision = 0.0
                else:
                    precision = (tp) / (tp + fp)

                if (tp + fn) == 0:
                    recall = 0.0
                else:
                    recall = (tp) / (tp + fn)

                if ((2 * tp) + tn + fp + fn) == 0:
                    fscore = 0.0
                else:
                    fscore = (2 * tp) / ((2 * tp) + tn + fp + fn)

                atbt_end = time.time()

                print(f'\n[{clf.upper()}] {mod.upper()} : {atbt.upper()} ({smpl.upper()})')
                print('--------------------')
                print(f'ACCURACY:  \t{accuracy}')
                print(f'F-SCORE:   \t{fscore}')
                print(f'PRECISION: \t{precision}')
                print(f'RECALL:    \t{recall}')
                print('--------------------')
                print(f'[{round(atbt_end - atbt_start, 2)} sec]')

                if smpl == 'non-smote':
                    with open(f'./results/{clf}_results.csv', 'a') as file:
                        file.write(f'{mod},{atbt},{smpl},accuracy,{accuracy}\n')
                        file.write(f'{mod},{atbt},{smpl},fscore,{fscore}\n')
                        file.write(f'{mod},{atbt},{smpl},precision,{precision}\n')
                        file.write(f'{mod},{atbt},{smpl},recall,{recall}\n')
                elif smpl == 'smote':
                    with open(f'./results/{clf}_smote_results.csv', 'a') as file:
                        file.write(f'{mod},{atbt},{smpl},accuracy,{accuracy}\n')
                        file.write(f'{mod},{atbt},{smpl},fscore,{fscore}\n')
                        file.write(f'{mod},{atbt},{smpl},precision,{precision}\n')
                        file.write(f'{mod},{atbt},{smpl},recall,{recall}\n')

            mod_end = time.time()

    all_end = time.time()

    print(f'\n[{clf.upper()}] ==> TOTAL TIME: {round(all_end - all_start, 2)} sec\n')