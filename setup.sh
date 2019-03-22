#!/bin/bash

if [ -d data ]
    then rm -rf data
fi

unzip data.zip -d data

echo '\n-- split.py --\n\n'; python split.py
echo '\n-- extract_dates.py --\n\n'; python extract_dates.py
echo '\n-- extract_activity.py --\n\n'; python extract_activity.py
echo '\n-- join.py --\n\n'; python join.py