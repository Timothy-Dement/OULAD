#!/bin/bash


# check for zip, folder first
# delete if present
# download
# unzip

curl -o data.zip https://analyse.kmi.open.ac.uk/open_dataset/download
unzip data.zip -d data

# Run preprocessing scripts