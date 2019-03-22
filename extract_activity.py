import os
import numpy as np
import pandas as pd
import sys
import time

vle = pd.read_csv('./data/vle.csv')
activity_types = vle['activity_type'].unique()

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

for mod in modules:

    print(f'Extracting {mod.upper()} VLE activity ...')

    mod_start = time.time()

    mod_assessments = pd.read_csv(f'./data/{mod}/{mod}_assessments_extracted.csv')
    mod_student_assessment = pd.read_csv(f'./data/{mod}/{mod}_student_assessment.csv')

    mod_vle = pd.read_csv(f'./data/{mod}/{mod}_vle.csv')
    mod_student_vle = pd.read_csv(f'./data/{mod}/{mod}_student_vle.csv')

    mod_student_vle = pd.merge(mod_student_vle, mod_vle[['id_site', 'activity_type']], on=['id_site'])

    mod_student_assessment = mod_student_assessment[mod_student_assessment['id_assessment'].isin(mod_assessments['id_assessment'])]
    mod_student_assessment = pd.merge(mod_student_assessment, mod_assessments, on=['id_assessment'])

    mod_student_assessment['due_vs_submission_date'] = mod_student_assessment['date'] - mod_student_assessment['date_submitted']

    for activity_type in activity_types:

        mod_student_assessment[f'{activity_type}_clicks_by_days'] = 0
        mod_student_assessment[f'{activity_type}_clicks_by_days_change'] = 0

        mod_student_assessment[f'{activity_type}_clicks_by_interval'] = 0
        mod_student_assessment[f'{activity_type}_clicks_by_interval_change'] = 0

    for index, row in mod_student_assessment.iterrows():

        num_items = len(mod_student_assessment.index)
        print(f'\t... {mod} {round((index / num_items) * 100, 2)}% complete')
        
        student_interactions = mod_student_vle[(mod_student_vle['id_student'] == row['id_student']) &
                                               (mod_student_vle['code_presentation'] == row['code_presentation'])]

        by_days_interactions = student_interactions[(student_interactions['date'] <= row['date']) &
                                                    (student_interactions['date'] > row['days_start'])]

        previous_by_days_interactions = None

        if np.isfinite(row['previous_days_start']) and np.isfinite(row['previous_days_end']):
            previous_by_days_interactions = student_interactions[(student_interactions['date'] <= row['previous_days_end']) &
                                                                 (student_interactions['date'] > row['previous_days_start'])]

        by_interval_interactions = student_interactions[(student_interactions['date'] <= row['date']) &
                                                        (student_interactions['date'] > row['interval_start'])]

        previous_by_interval_interactions = None

        if np.isfinite(row['previous_interval_start']) and np.isfinite(row['previous_interval_end']):
            previous_by_interval_interactions = student_interactions[(student_interactions['date'] <= row['previous_interval_end']) &
                                                                     (student_interactions['date'] > row['previous_interval_start'])]

        for activity_type in activity_types:

            type_by_days = by_days_interactions[by_days_interactions['activity_type'] == activity_type]

            type_previous_by_days = None

            if previous_by_days_interactions is not None:
                type_previous_by_days = previous_by_days_interactions[previous_by_days_interactions['activity_type'] == activity_type]

            type_by_interval = by_interval_interactions[by_interval_interactions['activity_type'] == activity_type]

            type_previous_by_interval = None

            if previous_by_interval_interactions is not None:
                type_previous_by_interval = previous_by_interval_interactions[previous_by_interval_interactions['activity_type'] == activity_type]

            by_days_count = type_by_days['sum_click'].sum()

            if type_previous_by_days is not None:
                previous_by_days_count = type_previous_by_days['sum_click'].sum()
            else:
                previous_by_days_count = 0

            by_interval_count = type_by_interval['sum_click'].sum()

            if type_previous_by_interval is not None:
                previous_by_interval_count = type_previous_by_interval['sum_click'].sum()
            else:
                previous_by_interval_count = 0

            change_by_days = by_days_count - previous_by_days_count
            change_by_interval = by_interval_count - previous_by_interval_count

            mod_student_assessment.at[index, f'{activity_type}_clicks_by_days'] = by_days_count
            mod_student_assessment.at[index, f'{activity_type}_clicks_by_days_change'] = change_by_days
            mod_student_assessment.at[index, f'{activity_type}_clicks_by_interval'] = by_interval_count
            mod_student_assessment.at[index, f'{activity_type}_clicks_by_interval_change'] = change_by_interval

    mod_student_assessment.to_csv(f'./data/{mod}/{mod}_student_assessment_extracted.csv', index=False)

    mod_end = time.time()

    print(f'DONE [{round(mod_end - mod_start, 2)} sec]')
