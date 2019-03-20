import math
import multiprocessing
import pandas
import seaborn
import time

from imblearn.over_sampling import SMOTE

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

total_start_time = time.time()

student_info = pandas.read_csv('./data/studentInfo.csv')

student_data = student_info.drop(columns=['id_student'])

# student_data = student_data.drop(columns=['code_module', 'code_presentation'])
# student_data = student_data.drop(columns=['gender', 'region', 'highest_education', 'imd_band', 'disability'])
# student_data = student_data.drop(columns=['num_of_prev_attempts', 'studied_credits'])

student_data['final_result'] = student_data['final_result'].apply(lambda x: 'AR' if (x == 'Withdrawn' or x == 'Fail') else 'NAR')

le = preprocessing.LabelEncoder()

if 'imd_band' in list(student_data):
        student_data['imd_band'] = student_data['imd_band'].apply(lambda x: x if type(x) == str else 'blank')

for attribute in list(student_data):
    if student_data[attribute].dtypes == 'object' and attribute != 'final_result':
        student_data[attribute] = le.fit_transform(student_data[attribute])

# rf = RandomForestClassifier(random_state=0, n_estimators=10000, max_depth=int(len(list(student_data)) ** 0.5))
rf = RandomForestClassifier(random_state=0, n_estimators=10000, max_depth=int(len(list(student_data)) ** 0.5), n_jobs=-1)

sk = StratifiedKFold(random_state=0, n_splits=10, shuffle=True)

metrics = { 'acc': [], 'prec': [], 'rec': [], 'f_score': [] }

labels = student_data['final_result']
unlabeled_data = student_data.drop(columns=['final_result'])

fold = 0

for train_indices, test_indices in sk.split(unlabeled_data, labels):

    fold_start_time = time.time()

    rf_clone = clone(rf)

    train_data = unlabeled_data.iloc[train_indices]
    train_labels = labels.iloc[train_indices]

    test_data = unlabeled_data.iloc[test_indices]
    test_labels = labels.iloc[test_indices]

    rf_clone.fit(train_data, train_labels)

    predictions = rf_clone.predict(test_data)

    fold_end_time = time.time()

    prec, rec, _, _ = precision_recall_fscore_support(test_labels, predictions, pos_label='AR', average='binary')
    acc = accuracy_score(test_labels, predictions)

    print(f'AR ACCURACY:  \t{acc}')
    print(f'AR PRECISION: \t{prec}')
    print(f'AR RECALL:    \t{rec}\n')

total_end_time = time.time()

print(f'\n\n\n\n\nTIME: {total_end_time - total_start_time}')