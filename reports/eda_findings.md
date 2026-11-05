# EDA

## Dataset Overview
- Dataset contains 79,330 bookings and 31 columns.
- Target variables: ADR (regression) and IsCanceled (classification).

## Missing Values
- Children: 4 missing values.
- Country: 24 missing values.
- Agent: 8,131 literal NULL values.
- Company: 75,641 literal NULL values.

No missing values have been filled during EDA. Missing-value treatment will be handled during preprocessing.

## Duplicate Analysis
- 25,902 duplicate rows (32.65%).
- Duplicate records have been retained because identical booking attributes do not necessarily represent the same reservation.

## Invalid Values
- No negative values were found in the seven numerical columns checked.
- 167 bookings contain zero recorded guests.
- An optional cleaning function was implemented to exclude these records if justified.

## Outlier Analysis
Potential outliers were identified using the 1.5 × IQR method.

| Feature | Outliers |
|---|---:|
| ADR | 3,387 |
| LeadTime | 2,576 |
| StaysInWeekendNights | 82 |
| StaysInWeekNights | 719 |

Outliers were retained pending further investigation.

## Visualization Findings
- ADR is right-skewed, with most bookings concentrated at lower daily rates.
- LeadTime is right-skewed.
- Canceled bookings generally have longer lead times than non-canceled bookings.
- Approximately 41.73% of bookings were canceled.

## Cleaning Decisions
- Original dataset preserved.
- Column names stripped of surrounding whitespace.
- Optional exclusion of zero-guest bookings supported.
- Missing values, duplicates and outliers retained by default.
- Model-specific preprocessing decisions deferred to Member 2.