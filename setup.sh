#!/bin/bash

if [ -d data ]
    then rm -rf data
fi

unzip data.zip -d data

python split.py
python extract_assessments.py
# extract_student_assessment.py
# join.py