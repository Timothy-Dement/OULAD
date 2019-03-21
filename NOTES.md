# NOTES

- Classifiers used in Systematic Review:

    1. Neural Network
    2. Decision Tree
    3. Support Vector Machine
    4. K-Nearest Neighbor
    5. Naive Bayes

- Grouping attributes by type, testing combinations of attribute groups

- Clustering

- Change in click activity per assignment period (per Wolff)

- Try aggregate vs. separated-by-courses

- Missing due dates for some assessments (all are "Exams"):

| Code Modlue | Code Presentation | Id Assessment | Submissions |
| ----------- | ----------------- | ------------- | ----------- |
| AAA         | 2013J             | 1757          | no          |
| AAA         | 2014J             | 1763          | no          |
| BBB         | 2013B             | 14990         | no          |
| BBB         | 2013J             | 15002         | no          |
| BBB         | 2014B             | 15014         | no          |
| BBB         | 2014J             | 15025         | no          |
| CCC         | 2014B             | 24290         | **YES**     |
| CCC         | 2014B             | 40087         | no          |
| CCC         | 2014J             | 24299         | **YES**     |
| CCC         | 2014J             | 40088         | no          |
| DDD         | 2014J             | 25368         | **YES**     |

Only affects 1,915 out of 18,941 records in ccc_student_assessment.csv (~10%) and 951 out of 30,865 records in ddd_student_assessment.csv (~3%).

Overall, this affects 2,866 out of 173,913 records in the original studentAssessment.csv file (~1%).

# QUESTIONS

- Problem with training on future and testing on past?
- Train and test by presentation time?
- More by Annika Wolff?
- How should data be separated?
- Can we extract new features from VLE data?
- Normalization?
- Divide VLE interations into bins (assessment-date to assessment-date) or (x-days-before assessment-date)?

# THREATS TO VALIDITY

- No way to associate VLE resources with assessments with complete confidence (date assumption)
- Missing due dates for some assessments
- Assumes no overlap between assessments (possibly)
- Assumes VLE interaction starts at zero for first assessment
- Chose 2 weeks arbitrarily