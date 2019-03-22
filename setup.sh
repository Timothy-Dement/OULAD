#!/bin/bash

if [ -d data ]
    then rm -rf data
fi

unzip data.zip -d data

python split.py
python extract_dates.py
# extract_student_activity.py
# join.py