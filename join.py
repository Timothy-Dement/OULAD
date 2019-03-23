import os
import numpy as np
import pandas as pd
import sys
import time

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

activity_attributes = ['due_vs_submission_date']

activity_types = ['resource',
                  'oucontent',
                  'url',
                  'homepage',
                  'subpage',
                  'glossary',
                  'forumng',
                  'oucollaborate',
                  'dataplus',
                  'quiz',
                  'ouelluminate',
                  'sharedsubpage',
                  'questionnaire',
                  'page',
                  'externalquiz',
                  'ouwiki',
                  'dualpane',
                  'repeatactivity',
                  'folder',
                  'htmlactivity']

for activity_type in activity_types:
    activity_attributes.append(f'{activity_type}_clicks_by_days')
    activity_attributes.append(f'{activity_type}_clicks_by_days_change')
    activity_attributes.append(f'{activity_type}_clicks_by_interval')
    activity_attributes.append(f'{activity_type}_clicks_by_interval_change')

assessment_attributes = ['id_assessment',
                         'assessment_type',
                         'date',
                         'weight']

student_attributes = ['id_student',
                      'gender',
                      'region',
                      'highest_education',
                      'imd_band',
                      'age_band',
                      'num_of_prev_attempts',
                      'studied_credits',
                      'disability']

selected_attributes = ['code_module', 'code_presentation'] + student_attributes + assessment_attributes + activity_attributes + ['score']

all_composite = pd.DataFrame(columns=selected_attributes)

if not os.path.exists('./data/_composite'):
    os.mkdir('./data/_composite')

for mod in modules:

    print(f'Joining {mod.upper()} ...', end=' ')
    sys.stdout.flush()

    mod_start = time.time()

    mod_student_assessment_extracted = pd.read_csv(f'./data/{mod}/{mod}_student_assessment_extracted.csv')
    mod_student_info = pd.read_csv(f'./data/{mod}/{mod}_student_info.csv')

    mod_composite = pd.merge(mod_student_assessment_extracted, mod_student_info, on=['code_module', 'code_presentation', 'id_student'])

    mod_composite = mod_composite[selected_attributes]

    all_composite = all_composite.append(mod_composite, ignore_index=True)

    mod_composite.to_csv(f'./data/_composite/{mod}_composite.csv', index=False)

    mod_end = time.time()
    print(f'DONE [{round(mod_end - mod_start, 2)} sec]')

all_composite.to_csv('./data/_composite/zzz_composite.csv', index=False)
