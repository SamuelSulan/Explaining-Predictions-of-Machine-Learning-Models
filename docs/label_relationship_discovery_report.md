# MM-IMDb Label Relationship Discovery Report

## Purpose

This report explores relationships between the 23 MM-IMDb genre labels. The goal is to understand which genres commonly appear together, which genres are unusually strongly associated after correcting for their popularity, and which genres rarely appear together.

This matters because the dataset is highly imbalanced. For example, `Drama` appears in more than half of all movies, so raw co-occurrence counts involving `Drama` can look large even when the relationship is not especially meaningful. For that reason, this report uses scaled association metrics in addition to raw counts.

## Metrics Used

### Raw Co-occurrence Count

Raw co-occurrence counts how many movies contain both labels.

Example:

- `Drama + Romance = 3629`

This is useful, but it is biased toward common labels.

### Conditional Probability

Conditional probability measures how often one label appears when another label is present.

Example:

- `Film-Noir -> Crime = 71.6%`

This means that 71.6% of Film-Noir movies in the dataset are also labeled as Crime.

This metric is directional:

- `A -> B` can be very different from `B -> A`

### Lift

Lift compares observed co-occurrence against expected co-occurrence if the labels were independent.

Formula:

```text
lift(A, B) = P(A and B) / (P(A) * P(B))
```

Interpretation:

| Lift value | Meaning |
|---:|---|
| `> 1` | Labels appear together more often than expected by chance |
| `= 1` | Labels appear together about as often as expected |
| `< 1` | Labels appear together less often than expected |

Lift is useful for correcting popularity. A rare pair can have high lift if the two labels appear together much more often than their individual frequencies suggest.

### Phi Correlation

Phi correlation is a correlation coefficient for two binary labels.

Interpretation:

| Phi value | Meaning |
|---:|---|
| positive | Labels tend to appear together |
| near zero | Weak or no relationship |
| negative | Labels tend not to appear together |

Phi is useful because it scales association according to both positive and negative label counts. In this report, phi is the main balanced metric for identifying strong positive and negative label relationships.

### Jaccard Similarity

Jaccard similarity measures overlap between two label sets.

Formula:

```text
Jaccard(A, B) = count(A and B) / count(A or B)
```

It is useful for simple overlap interpretation, but it can still be affected by label frequency.

## Generated Figures

The following figures were generated for visual inspection:

- `outputs/figures/label_phi_correlation.png`
- `outputs/figures/label_lift_heatmap.png`
- `outputs/figures/label_conditional_probability.png`

The phi correlation heatmap is best for balanced positive and negative relationships. The lift heatmap is best for finding pairs that occur much more often than expected. The conditional probability heatmap is best for interpreting directional relationships such as "when label A appears, how often does label B also appear?"

## Strongest Raw Co-occurrences

These are the most common label pairs by raw count.

| Pair | Count | Jaccard | A -> B | B -> A |
|---|---:|---:|---:|---:|
| Drama + Romance | 3629 | 0.231 | 26.0% | 67.7% |
| Drama + Comedy | 3210 | 0.166 | 23.0% | 37.4% |
| Drama + Thriller | 2722 | 0.166 | 19.5% | 52.4% |
| Comedy + Romance | 2540 | 0.222 | 29.6% | 47.4% |
| Drama + Crime | 2422 | 0.157 | 17.3% | 63.1% |
| Thriller + Crime | 2075 | 0.298 | 40.0% | 54.1% |
| Thriller + Action | 1559 | 0.217 | 30.0% | 43.9% |
| Drama + Action | 1396 | 0.087 | 10.0% | 39.3% |
| Thriller + Mystery | 1293 | 0.217 | 24.9% | 62.9% |
| Thriller + Horror | 1196 | 0.179 | 23.0% | 44.2% |
| Action + Adventure | 1106 | 0.215 | 31.2% | 40.8% |
| Drama + War | 1068 | 0.075 | 7.6% | 80.0% |
| Crime + Action | 1063 | 0.168 | 27.7% | 29.9% |
| Drama + Mystery | 1060 | 0.071 | 7.6% | 51.5% |
| Comedy + Crime | 1009 | 0.088 | 11.7% | 26.3% |

Raw co-occurrence confirms that common genres dominate the dataset. `Drama` appears in many top pairs because it is the largest label overall.

## Strongest Pairs By Lift

The following pairs occur much more often than expected after scaling by individual label frequencies. Pairs with fewer than 50 co-occurrences were excluded to avoid very unstable rare-pair results.

| Pair | Count | Expected Count | Lift | Phi |
|---|---:|---:|---:|---:|
| Animation + Short | 185 | 18.1 | 10.23 | 0.251 |
| Family + Animation | 569 | 64.1 | 8.88 | 0.413 |
| War + History | 367 | 58.8 | 6.24 | 0.262 |
| Biography + History | 340 | 59.1 | 5.75 | 0.238 |
| Fantasy + Animation | 380 | 74.2 | 5.12 | 0.233 |
| Crime + Film-Noir | 242 | 50.0 | 4.84 | 0.184 |
| Fantasy + Family | 536 | 124.2 | 4.32 | 0.246 |
| Adventure + Animation | 435 | 104.1 | 4.18 | 0.217 |
| Animation + Musical | 131 | 32.3 | 4.06 | 0.112 |
| Adventure + Family | 701 | 174.1 | 4.03 | 0.271 |
| Family + Musical | 216 | 54.0 | 4.00 | 0.144 |
| Adventure + Fantasy | 707 | 201.8 | 3.50 | 0.242 |
| Biography + Music | 187 | 54.1 | 3.46 | 0.118 |
| Family + Short | 102 | 30.3 | 3.37 | 0.084 |
| Biography + Sport | 106 | 32.8 | 3.23 | 0.082 |

Key observations:

- `Animation + Short` has the highest lift, appearing more than 10 times more often than expected.
- `Family + Animation` is both high-lift and high-phi, meaning it is not only surprisingly frequent but also a strong balanced association.
- Historical and biographical genres form meaningful clusters: `War + History`, `Biography + History`.
- `Crime + Film-Noir` is a strong scaled relationship, even though Film-Noir is rare.

## Strongest Positive Relationships By Phi Correlation

Phi correlation is the best balanced indicator of labels that truly tend to appear together.

| Pair | Count | Expected Count | Lift | Phi |
|---|---:|---:|---:|---:|
| Family + Animation | 569 | 64.1 | 8.88 | 0.413 |
| Thriller + Crime | 2075 | 767.6 | 2.70 | 0.355 |
| Thriller + Mystery | 1293 | 411.4 | 3.14 | 0.314 |
| Adventure + Family | 701 | 174.1 | 4.03 | 0.271 |
| Action + Adventure | 1106 | 370.6 | 2.98 | 0.270 |
| War + History | 367 | 58.8 | 6.24 | 0.262 |
| Animation + Short | 185 | 18.1 | 10.23 | 0.251 |
| Fantasy + Family | 536 | 124.2 | 4.32 | 0.246 |
| Adventure + Fantasy | 707 | 201.8 | 3.50 | 0.242 |
| Biography + History | 340 | 59.1 | 5.75 | 0.238 |
| Thriller + Action | 1559 | 710.0 | 2.20 | 0.238 |
| Fantasy + Animation | 380 | 74.2 | 5.12 | 0.233 |
| Adventure + Animation | 435 | 104.1 | 4.18 | 0.217 |
| Action + Sci-Fi | 782 | 272.3 | 2.87 | 0.215 |
| Thriller + Horror | 1196 | 540.6 | 2.21 | 0.207 |

Key clusters:

- family/animation/fantasy/adventure cluster
- thriller/crime/mystery/action cluster
- biography/history/war cluster
- action/sci-fi cluster

## Strongest Negative Relationships By Phi Correlation

These pairs appear together less often than expected. To avoid extremely rare-label noise, this table includes pairs where both labels have at least 300 total samples.

| Pair | Count | Expected Count | Lift | Phi |
|---|---:|---:|---:|---:|
| Drama + Documentary | 158 | 1120.2 | 0.14 | -0.274 |
| Drama + Horror | 523 | 1454.3 | 0.36 | -0.236 |
| Drama + Comedy | 3210 | 4622.8 | 0.69 | -0.232 |
| Comedy + Thriller | 585 | 1718.5 | 0.34 | -0.232 |
| Drama + Sci-Fi | 451 | 1071.2 | 0.42 | -0.180 |
| Comedy + Documentary | 172 | 689.1 | 0.25 | -0.156 |
| Romance + Documentary | 14 | 430.2 | 0.03 | -0.146 |
| Thriller + Documentary | 14 | 416.4 | 0.03 | -0.143 |
| Drama + Animation | 182 | 536.4 | 0.34 | -0.143 |
| Drama + Fantasy | 564 | 1040.0 | 0.54 | -0.140 |
| Romance + Horror | 117 | 558.5 | 0.21 | -0.138 |
| Drama + Family | 461 | 897.5 | 0.51 | -0.138 |
| Romance + Thriller | 499 | 1072.8 | 0.47 | -0.136 |
| Drama + Adventure | 918 | 1458.1 | 0.63 | -0.136 |
| Comedy + History | 72 | 378.3 | 0.19 | -0.122 |

Key observations:

- `Drama + Documentary` is the strongest negative pair after scaling.
- `Comedy + Thriller` is also strongly negative, even though both labels are individually common.
- `Romance` rarely appears with `Documentary`, `Horror`, and `Thriller`.
- Some raw counts can still be large but negatively associated. For example, `Drama + Comedy` appears 3210 times, but the expected count is 4622.8 because both labels are very common.

## Strongest Directional Relationships

Directional relationships answer this question:

```text
When label A appears, how often does label B also appear?
```

| Rule | Count | Source Count | Confidence | Lift |
|---|---:|---:|---:|---:|
| Film-Noir -> Drama | 281 | 338 | 83.1% | 1.55 |
| War -> Drama | 1068 | 1335 | 80.0% | 1.49 |
| Biography -> Drama | 986 | 1343 | 73.4% | 1.36 |
| History -> Drama | 831 | 1143 | 72.7% | 1.35 |
| Film-Noir -> Crime | 242 | 338 | 71.6% | 4.84 |
| Romance -> Drama | 3629 | 5364 | 67.7% | 1.26 |
| Crime -> Drama | 2422 | 3838 | 63.1% | 1.17 |
| Mystery -> Thriller | 1293 | 2057 | 62.9% | 3.14 |
| Musical -> Comedy | 516 | 841 | 61.4% | 1.85 |
| Film-Noir -> Thriller | 204 | 338 | 60.4% | 3.02 |
| Animation -> Family | 569 | 997 | 57.1% | 8.88 |
| Sport -> Drama | 350 | 634 | 55.2% | 1.03 |
| Family -> Comedy | 912 | 1668 | 54.7% | 1.65 |
| Crime -> Thriller | 2075 | 3838 | 54.1% | 2.70 |
| Music -> Drama | 557 | 1045 | 53.3% | 0.99 |
| Thriller -> Drama | 2722 | 5192 | 52.4% | 0.97 |
| Mystery -> Drama | 1060 | 2057 | 51.5% | 0.96 |
| Romance -> Comedy | 2540 | 5364 | 47.4% | 1.43 |
| Musical -> Romance | 381 | 841 | 45.3% | 2.19 |
| Horror -> Thriller | 1196 | 2703 | 44.2% | 2.21 |

Important interpretation detail:

- High confidence does not always mean a strong scaled relationship.
- For example, `War -> Drama` has high confidence because many War movies are also Drama, but its lift is only 1.49.
- `Film-Noir -> Crime` is more distinctive because it has both high confidence and high lift.

## Strongest Positive And Negative Pair Per Label

This table gives one strongest positive and one strongest negative scaled relationship for each label.

| Label | Strongest Positive Pair | Phi | Lift | Count | Strongest Negative Pair | Phi | Lift | Count |
|---|---|---:|---:|---:|---|---:|---:|---:|
| Drama | Romance | 0.142 | 1.26 | 3629 | Documentary | -0.274 | 0.14 | 158 |
| Comedy | Romance | 0.155 | 1.43 | 2540 | Drama | -0.232 | 0.69 | 3210 |
| Romance | Comedy | 0.155 | 1.43 | 2540 | Documentary | -0.146 | 0.03 | 14 |
| Thriller | Crime | 0.355 | 2.70 | 2075 | Comedy | -0.232 | 0.34 | 585 |
| Crime | Thriller | 0.355 | 2.70 | 2075 | Fantasy | -0.096 | 0.19 | 54 |
| Action | Adventure | 0.270 | 2.98 | 1106 | Drama | -0.116 | 0.73 | 1396 |
| Adventure | Family | 0.271 | 4.03 | 701 | Drama | -0.136 | 0.63 | 918 |
| Horror | Thriller | 0.207 | 2.21 | 1196 | Drama | -0.236 | 0.36 | 523 |
| Documentary | Biography | 0.148 | 3.15 | 339 | Drama | -0.274 | 0.14 | 158 |
| Mystery | Thriller | 0.314 | 3.14 | 1293 | Comedy | -0.105 | 0.49 | 334 |
| Sci-Fi | Action | 0.215 | 2.87 | 782 | Drama | -0.180 | 0.42 | 451 |
| Fantasy | Family | 0.246 | 4.32 | 536 | Drama | -0.140 | 0.54 | 564 |
| Family | Animation | 0.413 | 8.88 | 569 | Drama | -0.138 | 0.51 | 461 |
| Biography | History | 0.238 | 5.75 | 340 | Comedy | -0.121 | 0.27 | 118 |
| War | History | 0.262 | 6.24 | 367 | Comedy | -0.106 | 0.35 | 156 |
| History | War | 0.262 | 6.24 | 367 | Comedy | -0.122 | 0.19 | 72 |
| Music | Documentary | 0.134 | 3.21 | 269 | Thriller | -0.084 | 0.18 | 37 |
| Animation | Family | 0.413 | 8.88 | 569 | Drama | -0.143 | 0.34 | 182 |
| Musical | Family | 0.144 | 4.00 | 216 | Thriller | -0.082 | 0.10 | 17 |
| Western | Adventure | 0.049 | 1.86 | 137 | Drama | -0.066 | 0.63 | 240 |
| Sport | Biography | 0.082 | 3.23 | 106 | Thriller | -0.071 | 0.10 | 13 |
| Short | Animation | 0.251 | 10.23 | 185 | Drama | -0.102 | 0.30 | 77 |
| Film-Noir | Crime | 0.184 | 4.84 | 242 | Comedy | -0.081 | 0.00 | 0 |

## Main Dataset Discovery Conclusions

The dataset contains several clear genre communities:

- `Family`, `Animation`, `Fantasy`, and `Adventure` form a strong cluster.
- `Thriller`, `Crime`, `Mystery`, `Action`, and `Horror` form another strong cluster.
- `Biography`, `History`, and `War` form a historically oriented cluster.
- `Film-Noir` is rare but strongly connected to `Crime` and `Thriller`.

The strongest balanced positive relationship is:

- `Family + Animation`

The strongest high-lift relationship is:

- `Animation + Short`

The strongest negative relationship is:

- `Drama + Documentary`

The most important methodological point is that raw co-occurrence alone is not enough. Because some labels are much more frequent than others, scaled metrics such as lift and phi correlation give a clearer view of real genre relationships.

## How This Can Be Used In The Thesis

These findings can support several parts of the thesis:

- dataset analysis section
- motivation for multilabel classification
- motivation for per-label threshold tuning
- motivation for modeling label dependencies
- explanation of why macro F1 is preferred over only micro F1
- justification for classifier chains or neural label-correlation components
- qualitative discussion of model predictions and XAI outputs

For example, if the model predicts `Crime` together with `Thriller`, this is consistent with the dataset structure. If it predicts `Drama` together with `Documentary`, that combination is less typical according to the scaled label analysis.
