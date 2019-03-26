import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sample = 'non-smote'

for path in os.listdir(f'./dt-output/{sample}/'):

    title = None

    accuracy = []
    precision = []
    recall = []
    f_score = []

    with open (f'./dt-output/{sample}/{path}') as file:
        for index, line in enumerate(file):
            if index == 0:
                title = line
            else:
                if line.strip():
                    # depth = int(line.split('[')[1].split(']')[0])
                    metric = line.split(' ')[1].lower().replace(':', '')
                    value = float(line.split(':')[1].strip())

                    if metric == 'accuracy':
                        accuracy.append(value)
                    elif metric == 'precision':
                        precision.append(value)
                    elif metric == 'recall':
                        recall.append(value)
                    elif metric == 'f-score':
                        f_score.append(value)

    depth = list(range(1,11))

    df = pd.DataFrame()

    df['depth'] = depth
    df['accuracy'] = accuracy
    df['precision'] = precision
    df['recall'] = recall
    df['f_score'] = f_score

    sns.set()
    sns.lineplot(data=df[['accuracy', 'precision', 'recall', 'f_score']]).set_title(title)
    plt.show()
