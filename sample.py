import os
import numpy as np
import pandas as pd
import sys
import time

# `python sample.py {mod} {optional list of attribute subsets}`

# Running `python aaa sample.py` will by default include all attributes in the aaa data set.
# Individual subsets of attributes can be specified as command-line arguments.
# E.G., `python aa sample.py student assessment` will preserve only the student attributes and the assessment attributes in the aaa data set.
# We will likely only ever want to use one or the other of activity_days or activity_interval for comparison, but not both in one experiment.

options = {'student':           {'active': False,
                                 'attributes': ['gender',
                                                'region',
                                                'highest_education',
                                                'imd_band',
                                                'age_band',
                                                'num_of_prev_attempts',
                                                'studied_credits',
                                                'disability']},

           'assessment':        {'active': False,
                                 'attributes': ['assessment_type',
                                                'date',
                                                'weight']},

           'activity_days':     {'active': False,
                                 'attributes': ['due_vs_submission_date',
                                                'resource_clicks_by_days',
                                                'resource_clicks_by_days_change',
                                                'oucontent_clicks_by_days',
                                                'oucontent_clicks_by_days_change',
                                                'url_clicks_by_days',
                                                'url_clicks_by_days_change',
                                                'homepage_clicks_by_days',
                                                'homepage_clicks_by_days_change',
                                                'subpage_clicks_by_days',
                                                'subpage_clicks_by_days_change',
                                                'glossary_clicks_by_days',
                                                'glossary_clicks_by_days_change',
                                                'forumng_clicks_by_days',
                                                'forumng_clicks_by_days_change',
                                                'oucollaborate_clicks_by_days',
                                                'oucollaborate_clicks_by_days_change',
                                                'dataplus_clicks_by_days',
                                                'dataplus_clicks_by_days_change',
                                                'quiz_clicks_by_days',
                                                'quiz_clicks_by_days_change',
                                                'ouelluminate_clicks_by_days',
                                                'ouelluminate_clicks_by_days_change',
                                                'sharedsubpage_clicks_by_days',
                                                'sharedsubpage_clicks_by_days_change',
                                                'questionnaire_clicks_by_days',
                                                'questionnaire_clicks_by_days_change',
                                                'page_clicks_by_days',
                                                'page_clicks_by_days_change',
                                                'externalquiz_clicks_by_days',
                                                'externalquiz_clicks_by_days_change',
                                                'ouwiki_clicks_by_days',
                                                'ouwiki_clicks_by_days_change',
                                                'dualpane_clicks_by_days',
                                                'dualpane_clicks_by_days_change',
                                                'repeatactivity_clicks_by_days',
                                                'repeatactivity_clicks_by_days_change',
                                                'folder_clicks_by_days',
                                                'folder_clicks_by_days_change',
                                                'htmlactivity_clicks_by_days',
                                                'htmlactivity_clicks_by_days_change']},

           'activity_interval': {'active': False,
                                 'attributes': ['due_vs_submission_date',
                                                'resource_clicks_by_interval',
                                                'resource_clicks_by_interval_change',
                                                'oucontent_clicks_by_interval',
                                                'oucontent_clicks_by_interval_change',
                                                'url_clicks_by_interval',
                                                'url_clicks_by_interval_change',
                                                'homepage_clicks_by_interval',
                                                'homepage_clicks_by_interval_change',
                                                'subpage_clicks_by_interval',
                                                'subpage_clicks_by_interval_change',
                                                'glossary_clicks_by_interval',
                                                'glossary_clicks_by_interval_change',
                                                'forumng_clicks_by_interval',
                                                'forumng_clicks_by_interval_change',
                                                'oucollaborate_clicks_by_interval',
                                                'oucollaborate_clicks_by_interval_change',
                                                'dataplus_clicks_by_interval',
                                                'dataplus_clicks_by_interval_change',
                                                'quiz_clicks_by_interval',
                                                'quiz_clicks_by_interval_change',
                                                'ouelluminate_clicks_by_interval',
                                                'ouelluminate_clicks_by_interval_change',
                                                'sharedsubpage_clicks_by_interval',
                                                'sharedsubpage_clicks_by_interval_change',
                                                'questionnaire_clicks_by_interval',
                                                'questionnaire_clicks_by_interval_change',
                                                'page_clicks_by_interval',
                                                'page_clicks_by_interval_change',
                                                'externalquiz_clicks_by_interval',
                                                'externalquiz_clicks_by_interval_change',
                                                'ouwiki_clicks_by_interval',
                                                'ouwiki_clicks_by_interval_change',
                                                'dualpane_clicks_by_interval',
                                                'dualpane_clicks_by_interval_change',
                                                'repeatactivity_clicks_by_interval',
                                                'repeatactivity_clicks_by_interval_change',
                                                'folder_clicks_by_interval',
                                                'folder_clicks_by_interval_change',
                                                'htmlactivity_clicks_by_interval',
                                                'htmlactivity_clicks_by_interval_change']}}

mod = sys.argv[1]

for index, arg in enumerate(sys.argv):
    if len(sys.argv) > 2:
        if index > 1:
            options[arg]['active'] = True
    else:
        options['student']['active'] = True
        options['assessment']['active'] = True
        options['activity_days']['active'] = True
        options['activity_interval']['active'] = True

# We will not want to include these in model training and testing.
general_attributes = ['code_module', 'code_presentation', 'id_student', 'id_assessment']

selected_attributes = []

for item in options:
    if options[item]['active']:
        selected_attributes = selected_attributes + options[item]['attributes']

mod_frame = pd.read_csv(f'./data/_composite/{mod}_composite.csv')

mod_frame = mod_frame[selected_attributes + ['score']]

print(f'\nYou have chosen to include the following attributes for module {mod}:\n')

for item in list(mod_frame):
    print(f'\t - {item}')

print()