import pandas as pd

student_info = pd.read_csv('./data/studentInfo.csv')

print()

for header in list(student_info):
    if header == 'id_student':
        unique_values = pd.Series(student_info[header].unique())
        print(header, f'({len(unique_values)} unique)', '\n')
        for value in unique_values:
            print('\t', value)
        print()
