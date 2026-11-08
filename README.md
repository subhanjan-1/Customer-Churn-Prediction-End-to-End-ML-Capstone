# Customer Churn Prediction --- Day 16 Capstone

An end-to-end **tabular machine learning project** that predicts whether
a telecom-style customer is likely to leave a subscription service. The
project demonstrates the classical ML workflow: problem framing,
exploratory data analysis, leakage detection, preprocessing, model
comparison, hyperparameter tuning, evaluation, interpretation, and
business recommendations.

> **Project deliverable:** `day16_capstone.ipynb`\
> **Application type:** Jupyter Notebook --- this project does not
> require Flask, FastAPI, a web frontend, or deployment.

------------------------------------------------------------------------

## 1. Project objective

Customer churn occurs when a customer stops using a company's service.
Retaining existing customers is important for subscription businesses,
so the goal is to identify customers who may be at risk of leaving.

This project frames the task as a **binary classification problem**:

-   `0` --- customer stays
-   `1` --- customer churns

The model uses customer account and plan information available at
prediction time. The retention team can use the resulting risk scores to
decide which customers may need proactive support or retention offers.

### Business and model metrics

-   **Recall for the churn class:** the primary business metric to
    protect, because missing a customer who is about to leave may be
    costly.
-   **ROC-AUC:** the model-selection metric, used to compare how well
    models rank churners above non-churners.
-   **Precision and F1-score:** additional measures used to understand
    the quality of churn alerts and the precision--recall trade-off.

------------------------------------------------------------------------

## 2. Dataset

The notebook creates a synthetic telecom-style dataset directly in
memory using `make_classification` and NumPy. No external dataset
download is required.

-   **Number of records:** 4,000
-   **Target:** `churn`
-   **Expected churn proportion:** approximately 27% (may vary slightly
    with library versions or implementation details)
-   **Feature types:** numerical account features and categorical plan
    features
-   **Deliberate data-quality trap:** `exit_survey_sent`, a post-outcome
    field that can leak the target.

### Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `tenure_months`                     Number of months the customer has
                                      been with the service

  `monthly_charges`                   Customer's monthly bill

  `support_tickets`                   Number of support tickets

  `avg_latency_ms`                    Average service latency in
                                      milliseconds

  `addons_count`                      Number of add-on services

  `days_since_login`                  Days since the customer's last
                                      login

  `total_charges`                     Cumulative charges, calculated from
                                      monthly charges and tenure

  `contract_type`                     Month-to-month, one-year, or
                                      two-year contract

  `payment_method`                    Customer's payment method

  `internet_service`                  Type of internet service

  `has_dependents`                    Whether the customer has dependents

  `exit_survey_sent`                  Deliberate leakage feature: survey
                                      sent after churn

  `churn`                             Target label: 0 or 1
  -----------------------------------------------------------------------

**Important:** `exit_survey_sent` is removed before model training
because it represents information available only after the outcome.
`total_charges` is retained because it represents cumulative account
information available at prediction time in this synthetic exercise.

------------------------------------------------------------------------

## 3. Technology stack

-   Python
-   Jupyter Notebook / JupyterLab
-   NumPy and Pandas --- data generation and manipulation
-   Matplotlib --- visual exploration
-   Scikit-learn --- preprocessing, model training, cross-validation,
    tuning, and evaluation

No GPU, internet connection, paid API, or web application framework is
required.

------------------------------------------------------------------------

## 4. Project workflow --- step by step

The notebook is organised into the following eight steps.

### Step 1 --- Frame the problem and explore the data (EDA)

First, define the prediction task and identify the business metric. Then
inspect the dataset:

1.  Check the dataset shape and sample rows.
2.  Examine the distribution of the churn target.
3.  Compare numerical feature averages for churners and stayers.
4.  Calculate churn rates across contract types.
5.  Visualise tenure distributions and churn rates by contract.

**Why this matters:** EDA helps identify patterns and form hypotheses
before training models. For example, shorter tenure, more support
tickets, and month-to-month contracts may be associated with churn.

### Step 2 --- Split the data and detect leakage

Before modelling, identify columns that would not be available at
prediction time.

1.  Inspect the absolute correlation of numeric columns with `churn`.
2.  Flag suspiciously high correlations.
3.  Identify `exit_survey_sent` as a post-outcome leakage feature.
4.  Remove the leakage feature.
5.  Split the data into training, validation, and test sets using
    stratification.

The split is:

  Dataset        Share   Approximate records
  ------------ ------- ---------------------
  Training         70%                 2,800
  Validation       15%                   600
  Test             15%                   600

Stratification helps preserve a similar churn proportion in each split.

**Why this matters:** A model that uses information recorded after churn
can appear exceptionally accurate but will not be useful for predicting
future churn.

### Step 3 --- Preprocess features with a Pipeline

Numerical and categorical columns need different preprocessing.

-   Numerical features: `StandardScaler`
-   Categorical features: `OneHotEncoder(handle_unknown="ignore")`
-   Combined preprocessing: `ColumnTransformer`
-   End-to-end workflow: Scikit-learn `Pipeline`

The preprocessing is placed inside each model pipeline. This ensures
that transformations are fitted on the relevant training data during
cross-validation, rather than learning from validation or test data.

### Step 4 --- Train and compare four model families

Four classifiers are evaluated using the same five-fold stratified
cross-validation strategy and ROC-AUC scoring.

  -----------------------------------------------------------------------
  Model                               Purpose
  ----------------------------------- -----------------------------------
  Logistic Regression                 Linear classification baseline;
                                      also useful for interpretation

  Random Forest                       Ensemble of decision trees

  Gradient Boosting                   Sequential boosting of decision
                                      trees

  SVM with RBF kernel                 Non-linear, margin-based classifier
  -----------------------------------------------------------------------

The notebook records the mean and standard deviation of cross-validation
ROC-AUC for each model.

Gradient Boosting is selected for the tuning exercise, while Logistic
Regression is retained as an interpretable baseline.

### Step 5 --- Tune Gradient Boosting

`GridSearchCV` searches a small parameter grid using the same stratified
cross-validation strategy.

Parameters explored:

-   `n_estimators`: 150 and 300
-   `learning_rate`: 0.05 and 0.1
-   `max_depth`: 2 and 3

The best configuration is selected using cross-validation ROC-AUC. The
selected model is then checked against the validation set.

### Step 6 --- Evaluate the model and examine thresholds

The tuned model produces churn probabilities for the held-out test set.

The notebook evaluates:

-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   Confusion matrix

It also compares thresholds of 0.50, 0.40, 0.35, 0.30, and 0.25.

A lower threshold generally identifies more potential churners,
increasing recall but potentially reducing precision. This means the
retention team may catch more customers who are about to leave, while
also contacting more customers who would have stayed.

**Business consideration:** The final operating threshold should be
selected using validation data and the actual costs of missed churners
versus unnecessary retention offers. The notebook's threshold comparison
illustrates this trade-off; do not treat a test-set threshold sweep as
an independent threshold-selection procedure.

### Step 7 --- Interpret model predictions

The notebook uses two complementary approaches:

1.  **Permutation importance:** measures how much validation ROC-AUC
    changes when a feature is shuffled.
2.  **Logistic Regression coefficients:** show the direction of
    association learned by the interpretable baseline.

These techniques help explain which features contribute to predictions.
Importance and coefficients describe model behaviour; they do not
establish that a feature causes churn.

### Step 8 --- Prepare the business recommendation

The final step prints a scorecard containing:

-   Primary metric
-   Selected model
-   Test ROC-AUC
-   Operating threshold used in the illustration
-   Precision and recall at that threshold
-   Top permutation-importance features

The notebook also includes a manager-oriented recommendation and four
reflection questions covering leakage, metric choice, model trust, and
honest test-set usage.

------------------------------------------------------------------------

## 5. How to set up and run the project

### Prerequisites

Install Python 3.9 or later and Jupyter Notebook/JupyterLab. A CPU-based
environment is sufficient.

### Step 1 --- Create a project folder

For example:

``` text
customer-churn-capstone/
```

### Step 2 --- Place the notebook in the folder

``` text
customer-churn-capstone/
└── day16_capstone.ipynb
```

### Step 3 --- Create and activate a virtual environment (optional but recommended)

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4 --- Install the dependencies

``` bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

### Step 5 --- Start Jupyter

``` bash
jupyter notebook
```

Alternatively:

``` bash
jupyter lab
```

### Step 6 --- Open the notebook

Open `day16_capstone.ipynb` in the Jupyter interface.

### Step 7 --- Run all cells

Select:

**Kernel → Restart Kernel and Run All Cells**

Wait for execution to finish. Check that every cell runs without errors
and that the expected tables, metrics, and charts are displayed.

### Step 8 --- Save the completed notebook

Save the notebook with its executed outputs. Keep the required filename:

``` text
day16_capstone.ipynb
```

------------------------------------------------------------------------

## 6. Expected outputs

After running the notebook, you should see:

1.  A preview of the generated customer dataset.
2.  Class-balance and EDA summaries.
3.  Churn visualisations.
4.  Leakage-correlation results and split sizes.
5.  Preprocessing feature counts.
6.  Cross-validation ROC-AUC comparison for four models.
7.  Best Gradient Boosting parameters and validation ROC-AUC.
8.  Test classification report and confusion matrix.
9.  Precision/recall/F1 comparisons across thresholds.
10. Permutation importance and Logistic Regression coefficients.
11. A final scorecard, business recommendation, and reflection answers.

Exact scores can differ slightly across software versions. The important
requirement is that the workflow executes correctly and the results
support a reasoned conclusion.

------------------------------------------------------------------------

## 7. Important machine learning concepts demonstrated

  -----------------------------------------------------------------------
  Concept                             How it is used
  ----------------------------------- -----------------------------------
  Binary classification               Predicts churn or stay

  Class imbalance                     Churn is the minority class

  Stratified splitting                Preserves class proportions across
                                      splits

  Data leakage                        Detects and removes a post-outcome
                                      feature

  Feature scaling                     Standardises numerical features

  One-hot encoding                    Converts categorical features into
                                      numeric columns

  ColumnTransformer                   Applies different transformations
                                      to different columns

  Pipeline                            Combines preprocessing and model
                                      training

  Cross-validation                    Compares models across multiple
                                      folds

  GridSearchCV                        Searches for better hyperparameters

  Confusion matrix                    Counts true/false predictions

  Precision and recall                Measures alert quality and churn
                                      detection

  ROC-AUC                             Measures ranking performance across
                                      thresholds

  Permutation importance              Estimates feature importance by
                                      shuffling

  Model coefficients                  Provides signed effects for the
                                      Logistic Regression baseline
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 8. Limitations and responsible use

-   The dataset is synthetic and does not represent actual telecom
    customers.
-   Performance on this generated dataset does not guarantee performance
    on real business data.
-   Correlation and feature importance do not prove causation.
-   The cost of a false negative and a false positive should be
    estimated with the business team before choosing a production
    threshold.
-   A deployed system should be monitored for changing customer
    behaviour, data quality, and model performance.
-   Retention actions should be evaluated to determine whether the
    offers actually reduce churn.

------------------------------------------------------------------------

## 9. Final deliverable

The required capstone deliverable is:

``` text
day16_capstone.ipynb
```

The completed notebook should:

-   Run from a fresh kernel without errors.
-   Include outputs for the analysis and evaluation.
-   Contain the written recommendation.
-   Include answers to all four reflection questions.

This project is a notebook-based machine learning capstone. It does
**not** require a Flask application, REST API, frontend, or deployment.

------------------------------------------------------------------------

## 10. Project summary

This capstone demonstrates a complete classical machine learning
workflow for customer churn prediction. Starting with a synthetic
telecom-style customer table, it explores the data, identifies leakage,
builds a leakage-conscious preprocessing pipeline, compares four model
families, tunes Gradient Boosting, evaluates predictions using
classification metrics, interprets model behaviour, and translates the
findings into a business recommendation.

**The central objective is not simply to achieve a high score. It is to
build an honest, reproducible model evaluation workflow and communicate
how its predictions could support customer retention decisions.**
