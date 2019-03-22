#!/bin/bash

echo '';

if [ -d data ]
    then rm -rf data
fi

unzip data.zip -d data

echo ''; echo '-- split.py --'; echo ''; python split.py
echo ''; echo '-- extract_dates.py --'; echo ''; python extract_dates.py
echo ''; echo '-- extract_activity.py --'; echo ''; python extract_activity.py
echo ''; echo '-- join.py --'; echo ''; python join.py