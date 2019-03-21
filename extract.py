import numpy as np
import pandas as pd

modules = ['bbb']

for mod in modules:

    mod_assessments = pd.read_csv(f'./data/{mod}/{mod}_assessments.csv')
    mod_student_assessment = pd.read_csv(
        f'./data/{mod}/{mod}_student_assessment.csv')

    mod_vle = pd.read_csv(f'./data/{mod}/{mod}_vle.csv')
    mod_student_vle = pd.read_csv(f'./data/{mod}/{mod}_student_vle.csv')

    new_columns = pd.DataFrame(columns=['id_assessment',
                                        'days_start',
                                        'previous_days_start',
                                        'interval_start',
                                        'previous_interval_start'])

    presentations = mod_assessments['code_presentation'].unique()

    for pres in presentations:

        pres_assessments = mod_assessments[mod_assessments['code_presentation'] == pres]

        asmt_due_dates = pres_assessments[['id_assessment', 'date']]
        asmt_due_dates = asmt_due_dates.sort_values(by=['date'])
        asmt_due_dates = asmt_due_dates.reset_index(drop=True)

        asmt_due_dates['days_start'] = float('NaN')
        asmt_due_dates['previous_days_start'] = float('NaN')

        asmt_due_dates['interval_start'] = float('NaN')
        asmt_due_dates['previous_interval_start'] = float('NaN')

        num_days = 14

        for index, row in asmt_due_dates.iterrows():

            if np.isfinite(row['date']):

                if row['date'] - num_days > 0:
                    asmt_due_dates.at[index,
                                      'days_start'] = row['date'] - num_days
                else:
                    asmt_due_dates.at[index, 'days_start'] = 0

                if index > 0:
                    asmt_due_dates.at[index,
                                      'interval_start'] = asmt_due_dates.at[index - 1, 'date']
                else:
                    asmt_due_dates.at[index, 'interval_start'] = 0

                if index > 0:
                    asmt_due_dates.at[index,
                                      'previous_days_start'] = asmt_due_dates.at[index - 1, 'days_start']
                    asmt_due_dates.at[index,
                                      'previous_interval_start'] = asmt_due_dates.at[index - 1, 'interval_start']
                else:
                    asmt_due_dates.at[index, 'previous_days_start'] = 0
                    asmt_due_dates.at[index, 'previous_interval_start'] = 0

        new_columns = new_columns.append(asmt_due_dates.drop(columns=['date']), ignore_index=True)

    # print(mod_assessments)
    # print(new_columns)
    print(pd.merge(mod_assessments, new_columns, on=['id_assessment']))
