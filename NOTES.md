# NOTES

- 1 - Neural Network, 2 - Decision Tree, 3 - Support Vector Machine, 4 - K-Nearest Neighbor, 5 - Naive Bayes

- Grouping attributes by type, testing combinations of attribute groups

- Clustering

- Change in click activity per assignment period (per Wolff)

- Try aggregate vs. separated-by-courses

- Missing due dates for some assessments:

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


    - AAA:
        - 1757 - pres 2013J
        - 1763 - pres 2014J
    - BBB:
        - 14990 - pres 2013B
        - 15002 - pres 
        - 15014 - pres
        - 15025 - pres
    - CCC:
        - 24290 (HAS SUBMISSIONS) - pres
        - 40087 - pres
        - 24299 (HAS SUBMISSIONS) - pres
        - 40088 - pres
    - DDD:
        - 25368 (HAS SUBMISSIONS) - pres

    - Set this to highest number in date

# QUESTIONS

- Problem with training on future and testing on past?
- Train and test by presentation time?
- More by Annika Wolff?
- How should data be separated?
- Can we extract new features from VLE data?

# THREATS TO VALIDITY

- No way to associate VLE resources with assessments with complete confidence
- Missing dates for some assignments