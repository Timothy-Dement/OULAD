import os
import time

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(0)

# Record the overall start time
all_start = time.time()

# Attributes from studentInfo.csv
student_attributes = ['gender',
                      'region',
                      'highest_education',
                      'imd_band',
                      'age_band',
                      'num_of_prev_attempts',
                      'studied_credits',
                      'disability']

# Attributes from assessments.csv
assessment_attributes = ['assessment_type',
                         'date',
                         'weight']

# Attributes extracted from studentVle.csv (assessment period by days)
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

# Attributes extracted from studentVle.csv (assessment period by interval)
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

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

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

classifiers = ['ff']

if not os.path.exists('./nu_results'):
    os.mkdir('./nu_results')

for mod in modules:
    if os.path.exists(f'./nu_results/{mod}_ff-smote_results.csv'):
        os.remove(f'./nu_results/{mod}_ff-smote_results.csv')

for mod in modules:

    # Record start time for the module
    mod_start = time.time()

    # Read the appropriate CSV file into a dataframe
    mod_frame = pd.read_csv(f'./data/_composite/{mod}_composite.csv')

    # Drop rows containing one or more missing value
    mod_frame = mod_frame.dropna()

    # Drop columns irrelevant to classification
    mod_frame = mod_frame.drop(
        columns=['code_module', 'id_student', 'id_assessment'])

    # Fix typo in original data set for consistency
    mod_frame['imd_band'] = mod_frame['imd_band'].transform(
        lambda x: '10-20%' if x == '10-20' else x)

    # Fix columns that were cast as float64 to int64
    mod_frame['date'] = mod_frame['date'].astype(np.int64)
    mod_frame['due_vs_submission_date'] = mod_frame['due_vs_submission_date'].astype(
         np.int64)

    # Recast imd_band as an ordinal attribute
    mod_frame['imd_band'] = mod_frame['imd_band'].transform(
        lambda x: int(x.split('-')[1].split('%')[0]) / 100)

    # Recast age_band as an ordinal attribute
    mod_frame['age_band'] = mod_frame['age_band'].transform(
        lambda x: 0.0 if x == '0-35' else (0.5 if x == '35-55' else 1.0))

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
                mod_frame[column] = (mod_frame[column] - mod_frame[column].mean()) / mod_frame[column].std()
    
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
        X_train = train.drop(columns=['score'])
        X_test = test.drop(columns=['score'])

        # Cast score attribute as binary 'pass' (0) / 'fail' (1)
        y_train = train['score'].apply(lambda x: 0 if x >= 40 else 1)
        y_test = test['score'].apply(lambda x: 0 if x >= 40 else 1)

        # Apply SMOTE to the training data
        X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)

        for clf in classifiers:

            # Record start time for the classifier
            clf_start = time.time()

            ##################################################
            ##################################################

            model = Sequential()

            model.add(Dense(100, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
            model.add(Dense(75, kernel_initializer='normal', activation='relu'))
            model.add(Dense(100, kernel_initializer='normal', activation='relu'))
            model.add(Dense(75, kernel_initializer='normal', activation='relu'))
            model.add(Dense(100, kernel_initializer='normal', activation='relu'))      
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=10, batch_size=10)
            
            y_hat = model.predict(X_test)
            y_hat = [round(x[0]) for x in y_hat]

            ##################################################
            ##################################################

            # Generate the confusion matrix for the predictions
            tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_hat).ravel()

            # Calculate performance metrics from the confusion matrix
            accuracy = None
            precision = None
            recall = None
            fscore = None

            # Accuracy (account for divide-by-zero)
            if (tp + tn + fp + fn) == 0:
                accuracy = 0.0
            else:
                accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Precision (account for divide-by-zero)
            if (tp + fp) == 0:
                precision = 0.0
            else:
                precision = (tp) / (tp + fp)

            # Recall (account for divide-by-zero)
            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = (tp) / (tp + fn)

            # F-Score (account for divide-by-zero)
            if ((2 * tp) + tn + fp + fn) == 0:
                fscore = 0.0
            else:
                fscore = (2 * tp) / ((2 * tp) + tn + fp + fn)

            # Report progress and results
            print('\n[ {0} : {1} : {2} : SMOTE ]'.format(mod.upper(), atbt.upper(), clf.upper()))
            print('+----------------------+')
            print('| ACC:    \t{0:.4f} |'.format(accuracy))
            print('| FSCORE: \t{0:.4f} |'.format(fscore))
            print('| PREC:   \t{0:.4f} |'.format(precision))
            print('| REC:    \t{0:.4f} |'.format(recall))
            print('+----------------------+')

            # Record end time and report runtime for the classifier
            clf_end = time.time()
            clf_s = clf_end - clf_start
            clf_m = clf_s / 60
            clf_h = clf_m / 60
            print('\n\t( T {0} : {1} : {2} = {3:.2f} s / {4:.2f} m / {5:.2f} h )'.format(mod.upper(), atbt.upper(), clf.upper(), clf_s, clf_m, clf_h))

            # Write results to appropriate file
            if not os.path.exists(f'./nu_results/{mod}_ff-smote_results.csv'):
                with open(f'./nu_results/{mod}_ff-smote_results.csv', 'w') as file:
                    file.write(f'module,attributes,classifier,technique,metric,score\n')
            
            with open(f'./nu_results/{mod}_ff-smote_results.csv', 'a') as file:
                file.write(f'{mod},{atbt},{clf},smote,accuracy,{accuracy}\n')
                file.write(f'{mod},{atbt},{clf},smote,fscore,{fscore}\n')
                file.write(f'{mod},{atbt},{clf},smote,precision,{precision}\n')
                file.write(f'{mod},{atbt},{clf},smote,recall,{recall}\n')
        
        # Record end time and report runtime for the attribute subset
        atbt_end = time.time()
        atbt_s = atbt_end - atbt_start
        atbt_m = atbt_s / 60
        atbt_h = atbt_m / 60
        print('\n\t( T {0} : {1} = {2:.2f} s / {3:.2f} m / {4:.2f} h )'.format(mod.upper(), atbt.upper(), atbt_s, atbt_m, atbt_h))

    # Record end time and report runtime for for the module
    mod_end = time.time()
    mod_s = mod_end - mod_start
    mod_m = mod_s / 60
    mod_h = mod_m / 60
    print('\n\t( T {0} = {1:.2f} s / {2:.2f} m / {3:.2f} h )'.format(mod.upper(), mod_s, mod_m, mod_h))

# Record overall end time and report overall runtime
all_end = time.time()
all_s = all_end - all_start
all_m = all_s / 60
all_h = all_m / 60
print('\n\t( T = {0:.2f} s / {1:.2f} m / {2:.2f} h )'.format(all_s, all_m, all_h))
