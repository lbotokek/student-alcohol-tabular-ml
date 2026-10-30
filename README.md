# Predicting heavy weekend drinking from a student's profile

Can we tell which secondary school students drink heavily at the weekend from their social and family background alone? This repository is a small tabular machine learning study on that question, with scikit-learn and skrub.

Originally my part of a group project for the SY09 course (Data analysis and machine learning) at Université de Technologie de Compiègne; extended and translated for this repository.

## Question

The target is binary: `high = 1` when weekend alcohol use `Walc` is 4 or 5 on a 1 to 5 scale. That is 132 of 649 students, about **20% positives**. The features are the social, family and school profile (28 columns: age, parents' education and jobs, study time, going out, family relations, support, etc.). Alcohol variables and grades are left out, since they are the target or an outcome.

## Data

[Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from the UCI Machine Learning Repository, licensed CC BY 4.0. Survey of students in two Portuguese secondary schools:

- `data/student-por.csv`: Portuguese class, 649 students (main sample);
- `data/student-mat.csv`: maths class, 395 students (robustness check).

> P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J. Teixeira (Eds.), Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008, EUROSIS.

## Method

- **No leakage**: scaling and one-hot encoding live inside a scikit-learn `Pipeline`, so they are fitted on the training folds only.
- **Stratified 5-fold cross-validation** (fixed seed), so every fold keeps the 20% of positives.
- **A majority-class baseline**, which already gets about 80% accuracy. Because of the imbalance, models are compared on **ROC AUC** and **F1**, not accuracy.
- **Models**: LDA, QDA, logistic regression, KNN, then `HistGradientBoostingClassifier` in the same pipeline, with skrub's `TableVectorizer`, and with `skrub.tabular_pipeline("classifier")`.
- **Interpretation** with logistic regression odds ratios (on standardised features), cross-checked with the LDA coefficients.
- **Robustness checks**: another threshold (`Walc >= 3`), the other sample (maths), and the model without the `goout` feature.
- Exploratory part: χ² tests, PCA and a small k-means typology.

## Key results

Stratified 5-fold CV on the Portuguese sample (mean over folds):

| Model | Accuracy | F1 | ROC AUC | AUC std |
|---|---|---|---|---|
| LDA | 0.829 | 0.512 | 0.817 | 0.037 |
| Logistic regression | 0.840 | 0.521 | 0.816 | 0.035 |
| Gradient boosting (same preprocessing) | 0.817 | 0.467 | 0.806 | 0.031 |
| skrub `TableVectorizer` + gradient boosting | 0.817 | 0.467 | 0.806 | 0.031 |
| skrub `tabular_pipeline("classifier")` | 0.818 | 0.455 | 0.803 | 0.026 |
| QDA | 0.794 | 0.377 | 0.724 | 0.061 |
| KNN (k=15) | 0.803 | 0.118 | 0.717 | 0.044 |
| Baseline (majority class) | 0.797 | 0.000 | 0.500 | 0.000 |

- The linear models are the best here. Gradient boosting does not do better: its AUC is about 0.01 lower, which is within the fold-to-fold noise, and its F1 is lower. With 649 rows and a signal carried by a few main effects, that is not surprising.
- skrub gives the same scores as my hand-written preprocessing (trees do not care about scaling, and both one-hot encode the same columns), with less code to write and maintain.
- Strongest associations (logistic regression odds ratios): being male (4.55) and going out often (3.24, per standard deviation) raise the odds of heavy drinking; more study time (0.68) and better family relations (0.68) lower them.
- Robustness (logistic regression AUC): 0.800 on the maths sample, 0.735 with the `Walc >= 3` threshold, and 0.735 without `goout`. The signal holds on the other sample, and going out carries a large part of it.

## Limitations

- Small sample: 649 students and 132 positives, so differences of 0.01 AUC between models are not meaningful.
- Self-reported survey data from two schools in one country.
- The target depends on a threshold, and ordinal answers are treated as numbers.
- These are correlations, not causal effects: going out does not "cause" drinking in this analysis.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab notebooks/student_alcohol_analysis.ipynb
```

To re-execute the whole notebook from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/student_alcohol_analysis.ipynb
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the UCI Machine Learning Repository (Cortez and Silva, 2008).
