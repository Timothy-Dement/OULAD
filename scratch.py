import pandas as pd

modules = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

for module in modules:
    
    module_student_vle = pd.read_csv(f'./data/{module}/{module}_student_vle.csv')

    module_assessments = pd.read_csv(f'./data/{module}/{module}_assessments.csv')

    presentations = pd.Series(module_student_vle['code_presentation'].unique())

    for presentation in presentations:

        print(f'\n+-----+\n| {module} | -> {presentation}\n+-----+\n')

        presentation_student_vle = module_student_vle[module_student_vle['code_presentation'] == presentation]

        presentation_assessments = module_assessments[module_assessments['code_presentation'] == presentation]

        print(f"LAST VLE ACTION: {presentation_student_vle['date'].max()}")
        print(f"LAST ASSESSMENT: {presentation_assessments['date'].max()}")

