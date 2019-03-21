#!/bin/bash

if [ -f data.zip ]
    then rm -rf data.zip
fi

if [ -d data ]
    then rm -rf data
fi

curl -o data.zip https://analyse.kmi.open.ac.uk/open_dataset/download
unzip data.zip -d data

python split.py
python extract_assessments.py
# extract_student_assessment.py
# join.py