# CSC 522-001: Automated Learning and Data Analysis

- Python 3.6.8
- Pandas 0.24.1
- Numpy 1.15.4

To run data preprocessing (not recommended due to length):
```bash
git clone https://github.com/timothy-dement/OULAD.git
cd OULAD
. setup.sh
```

## Description of Attributes

| Attribute | Source | Description | Type |
| :---: | :---: | :---: | :---: |
| `code_module` | From original set | Code name of the module, which serves as the identifier. | Categorical |
| `code_presentation` | From original set | Code name of the presentation. It consists of the year and B for the presentation starting in February and J for the presentation starting in October. | Categorical |
| `id_student` | From original set | A unique identification number for the student. | Categorical |
| `gender` | From original set | The student's gender. | Categorical |
| `region` | From original set | Identifies the geographic region where the student lived while taking the module-presentation. | Categorical |
| `highest_education` | From original set | Highest student education level on entry to the module presentation. | Categorical |
| `imd_band` | From original set | Specifies the Index of Multiple Depravation band of the place where the student lived during the module-presentation. | Categorical |
| `age_band` | From original set | Band of the student's age. | Categorical |
| `num_of_prev_attempts` | From original set | The number of times the student has attempted this module. | Numerical |
| `studied_credits` | From original set | The total number of credits for the modules the student is currently studying. | Numerical |
| `disability` | From original set | Indicates whether the student has declared a disability. | Categorical |
| `id_assessment` | From original set | The identification number of the assessment. | Categorical |
| `assessment_type` | From original set | Type of assessment. Three types of assessments exits: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA), and Final Exam (Exam). | Categorical |
| `date` | From original set | Information about the final submission date of the assessment calculated as the number of days since the start of the module-presentation. The starting date of the presentation has number 0 (zero). | Numerical |
| `weight` | From original set | Weight of the assessment in %. Typically, Exams are treated separately and have the weight 100%; the sum of all other assessments is 100%. | Numerical |
| `due_vs_submission_date` | Extracted from data | The difference between the due date of a given assessment and the date the given student submitted the assignment. | Numerical |
| `resource_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `resource_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `resource_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `resource_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `oucontent_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `oucontent_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `oucontent_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `oucontent_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `url_clicks_by_days` | Extracted from data | See **Note 1** below. |
| `url_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `url_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `url_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `homepage_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `homepage_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `homepage_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `homepage_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `subpage_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `subpage_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `subpage_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `subpage_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `glossary_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `glossary_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `glossary_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `glossary_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `forumng_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `forumng_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `forumng_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `forumng_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `oucollaborate_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `oucollaborate_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `oucollaborate_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `oucollaborate_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `dataplus_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `dataplus_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `dataplus_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `dataplus_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `quiz_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `quiz_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `quiz_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `quiz_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `ouelluminate_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `ouelluminate_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `ouelluminate_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `ouelluminate_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `sharedsubpage_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `sharedsubpage_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `sharedsubpage_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `sharedsubpage_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `questionnaire_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `questionnaire_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `questionnaire_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `questionnaire_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `page_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `page_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `page_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `page_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `externalquiz_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `externalquiz_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `externalquiz_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `externalquiz_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `ouwiki_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `ouwiki_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `ouwiki_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `ouwiki_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `dualpane_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `dualpane_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `dualpane_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `dualpane_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `repeatactivity_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `repeatactivity_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `repeatactivity_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `repeatactivity_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `folder_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `folder_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `folder_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `folder_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `htmlactivity_clicks_by_days` | Extracted from data | See **Note 1** below. | Numerical |
| `htmlactivity_clicks_by_days_change` | Extracted from data | See **Note 2** below. | Numerical |
| `htmlactivity_clicks_by_interval` | Extracted from data | See **Note 3** below. | Numerical |
| `htmlactivity_clicks_by_interval_change` | Extracted from data | See **Note 4** below. | Numerical |
| `score` | From original set | The student's score in this assessment. The range is from 0 to 100. A score lower than 40 is interpreted as Fail. | Numerical |

**Note 1:**

- Attribute names of the format `{resource_type}_clicks_by_days` refer to the number of times the given student clicked the given resource type in the 14 days prior to the given assessment's due date ("by days").

**Note 2:**

- Attribute names of the format `{resource_type}_clicks_by_days_change` refer to the change in the number of times the given student clicked the given resource type between the preceeding assessment period and the current assessment period. Here, an assessment period refers to the 14 days prior to each assessment's due date ("by days").

**Note 3:**

- Attribute names of the format `{resource_type}_clicks_by_interval` refer to the number of times the given student clicked the given resource type in the period between the previous assessment due date and the current assessment due date ("by interval").

**Note 4:**

- Attribute names of the format `{resource_type}_clicks_by_interval_change` refer to the change in the number of times the given student clicked the given resource type between the preceeding assessment period and the current assessment period. Here, an assessment period refers to the period between the previous assessment due date and the current assessment due date ("by interval").