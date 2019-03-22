# CSC 522-001: Automated Learning and Data Analysis

Python 3.6.8
Pandas 0.24.1
Numpy 1.15.4

To run data preprocessing (not recommended due to length):
```bash
git clone https://github.com/timothy-dement/OULAD.git
cd OULAD
. setup.sh
```

## Description of Attributes

| Attribute | Source | Description |
| --- | --- | --- |
| `code_module` | From original set | Code name of the modules, which serves as the identifier. |
| `code_presentation` | From original set | Code name of the presentation. It consists of the year and B for the presentation starting in February and J for the presentation starting in October. |
| `id_student` | From original set | A unique identification number for the student. |
| `gender` | From original set | The student's gender. |
| `region` | From original set | Identifies the geographic region where the student lived while taking the module-presentation. |
| `highest_education` | From original set | Highest student education level on entry to the module presentation. |
| `imd_band` | From original set | Specifies the Index of Multiple Depravation band of the place where the student lived during the module-presentation. |
| `age_band` | From original set | Band of the student's age. |
| `num_of_prev_attempts` | From original set | The number of times the student has attempted this module. |
| `studied_credits` | From original set | The total number of credits for the modules the student is currently studying. |
| `disability` | From original set | Indicates whether the student has declared a disability. |
| `id_assessment` | From original set | The identification number of the assessment. |
| `assessment_type` | From original set | Type of assessment. Three tpes of assessments exits: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA), and Final Exam (Exam). |
| `date` | From original set | Information about the final submission date of the assessment calculated as the number of days since the start of the module-presentation. The starting date of the presentation has number 0 (zero). |
| `weight` | From original set | Weight of the assessment in %. Typically, Exams are treated separately and have the weight 100%; the sum of all other assessments is 100%. |
| `due_vs_submission_date` | Extracted from data | The difference between the due date of a given assessment and the date the given student submitted the assignment. |
| `resource_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `resource_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `resource_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `resource_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `oucontent_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `oucontent_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `oucontent_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `oucontent_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `url_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `url_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `url_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `url_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `homepage_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `homepage_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `homepage_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `homepage_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `subpage_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `subpage_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `subpage_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `subpage_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `glossary_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `glossary_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `glossary_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `glossary_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `forumng_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `forumng_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `forumng_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `forumng_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `oucollaborate_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `oucollaborate_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `oucollaborate_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `oucollaborate_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `dataplus_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `dataplus_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `dataplus_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `dataplus_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `quiz_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `quiz_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `quiz_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `quiz_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `ouelluminate_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `ouelluminate_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `ouelluminate_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `ouelluminate_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `sharedsubpage_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `sharedsubpage_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `sharedsubpage_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `sharedsubpage_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `questionnaire_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `questionnaire_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `questionnaire_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `questionnaire_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `page_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `page_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `page_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `page_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `externalquiz_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `externalquiz_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `externalquiz_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `externalquiz_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `ouwiki_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `ouwiki_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `ouwiki_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `ouwiki_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `dualpane_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `dualpane_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `dualpane_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `dualpane_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `repeatactivity_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `repeatactivity_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `repeatactivity_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `repeatactivity_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `folder_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `folder_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `folder_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `folder_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `htmlactivity_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `htmlactivity_clicks_by_days_change` | Extracted from data | See **Note 2** below. |
| `htmlactivity_clicks_by_interval` | Extracted from data | See **Note 3** below. |
| `htmlactivity_clicks_by_interval_change` | Extracted from data | See **Note 4** below. |
| `score` | From original set | The student's score in this assessment. The range is from 0 to 100. A score lower than 40 is interpreted as Fail. |

**Note 1:**

- Attribute names of the format `{resource_type}_clicks_by_days` refer to the number of times the given student clicked the given resource type in the 14 days prior to the given assessment's due date ("by days").

**Note 2:**

- Attribute names of the format `{resource_type}_clicks_by_days_change` refer to the change in the number of times the given student clicked the given resource type between the preceeding assessment period and the current assessment period. Here, an assessment period refers to the 14 days prior to each assessment's due date ("by days").

**Note 3:**

- Attribute names of the format `{resource_type}_clicks_by_interval` refer to the number of times the given student clicked the given resource type in the period between the previous assessment due date and the current assessment due date ("by interval").

**Note 4:**

- Attribute names of the format `{resource_type}_clicks_by_interval_change` refer to the change in the number of times the given student clicked the given resource type between the preceeding assessment period and the current assessment period. Here, an assessment period refers to the period between the previous assessment due date and the current assessment due date ("by interval").