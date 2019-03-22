import os
import pandas as pd
import time

assessments = pd.read_csv('./data/assessments.csv')
courses = pd.read_csv('./data/courses.csv')
student_assessment = pd.read_csv('./data/studentAssessment.csv')
student_info = pd.read_csv('./data/studentInfo.csv')
student_registration = pd.read_csv('./data/studentRegistration.csv')
student_vle = pd.read_csv('./data/studentVle.csv')
vle = pd.read_csv('./data/vle.csv')

modules = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']

for mod in modules:

    mod_assessments = assessments[assessments['code_module'] == mod]
    mod_courses = courses[courses['code_module'] == mod]
    mod_student_assessment = student_assessment[student_assessment['id_assessment'].isin(mod_assessments['id_assessment'])]
    mod_student_info = student_info[student_info['code_module'] == mod]
    mod_student_registration = student_registration[student_registration['code_module'] == mod]
    mod_student_vle = student_vle[student_vle['code_module'] == mod]
    mod_vle = vle[vle['code_module'] == mod]

    if not os.path.exists(f'./data/{mod.lower()}'):
        os.mkdir(f'./data/{mod.lower()}')

    mod_assessments.to_csv(f'./data/{mod.lower()}/{mod.lower()}_assessments.csv', index=False)
    mod_courses.to_csv(f'./data/{mod.lower()}/{mod.lower()}_courses.csv', index=False)
    mod_student_assessment.to_csv(f'./data/{mod.lower()}/{mod.lower()}_student_assessment.csv', index=False)
    mod_student_info.to_csv(f'./data/{mod.lower()}/{mod.lower()}_student_info.csv', index=False)
    mod_student_registration.to_csv(f'./data/{mod.lower()}/{mod.lower()}_student_registration.csv', index=False)
    mod_student_vle.to_csv(f'./data/{mod.lower()}/{mod.lower()}_student_vle.csv', index=False)
    mod_vle.to_csv(f'./data/{mod.lower()}/{mod.lower()}_vle.csv', index=False)
