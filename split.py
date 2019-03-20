import os
import pandas as pd

assessments = pd.read_csv('./data/assessments.csv')
courses = pd.read_csv('./data/courses.csv')
student_assessment = pd.read_csv('./data/studentAssessment.csv')
student_info = pd.read_csv('./data/studentInfo.csv')
student_registration = pd.read_csv('./data/studentRegistration.csv')
student_vle = pd.read_csv('./data/studentVle.csv')
vle = pd.read_csv('./data/vle.csv')

modules = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']

for module in modules:

    print(f'\n+-----+\n| {module} |\n+-----+\n')

    module_assessments = assessments[assessments['code_module'] == module]
    module_courses = courses[courses['code_module'] == module]
    module_student_assessment = student_assessment[student_assessment['id_assessment'].isin(module_assessments['id_assessment'])]
    module_student_info = student_info[student_info['code_module'] == module]
    module_student_registration = student_registration[student_registration['code_module'] == module]
    module_student_vle = student_vle[student_vle['code_module'] == module]
    module_vle = vle[vle['code_module'] == module]

    print(f'{module}-Assessments:         \t{len(module_assessments.index)}')
    print(f'{module}-Courses:             \t{len(module_courses.index)}')
    print(f'{module}-StudentAssessment:   \t{len(module_student_assessment.index)}')
    print(f'{module}-StudentInfo:         \t{len(module_student_info.index)}')
    print(f'{module}-StudentRegistration: \t{len(module_student_registration.index)}')
    print(f'{module}-StudentVle:          \t{len(module_student_vle.index)}')
    print(f'{module}-Vle:                 \t{len(module_vle.index)}\n')

    if not os.path.exists(f'./data/{module.lower()}'):
        os.mkdir(f'./data/{module.lower()}')

    module_assessments.to_csv(f'./data/{module.lower()}/{module.lower()}_assessments.csv', index=False)
    module_courses.to_csv(f'./data/{module.lower()}/{module.lower()}_courses.csv', index=False)
    module_student_assessment.to_csv(f'./data/{module.lower()}/{module.lower()}_student_assessment.csv', index=False)
    module_student_info.to_csv(f'./data/{module.lower()}/{module.lower()}_student_info.csv', index=False)
    module_student_registration.to_csv(f'./data/{module.lower()}/{module.lower()}_student_registration.csv', index=False)
    module_student_vle.to_csv(f'./data/{module.lower()}/{module.lower()}_student_vle.csv', index=False)
    module_vle.to_csv(f'./data/{module.lower()}/{module.lower()}_vle.csv', index=False)
