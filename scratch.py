import pandas as pd

student_vle = pd.read_csv('./data/studentVle.csv')

count = 0

for val in student_vle['date']:
    if val == 0:
        count += 1

print(count)