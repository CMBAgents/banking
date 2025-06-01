<!-- filename: bank_marketing_dataset_guide.md -->
This document provides a comprehensive guide for utilizing the preprocessed bank marketing dataset. It details the data loading procedure, explains each feature, discusses their potential influence on term deposit subscriptions based on Exploratory Data Analysis (EDA), and offers recommendations for subsequent modeling efforts.

## Loading the Processed Data

The preprocessing steps (detailed in Step 3) have generated two pandas DataFrames, `processed_train_df.pkl` and `processed_test_df.pkl`, which are ready for model training and evaluation. These files can be loaded into a Python environment using the pandas library as follows:

    import pandas as pd

    # Define the paths to the saved pickle files
    # These paths are based on the output directory from previous steps.
    # Adjust if your files are located elsewhere.
    output_data_dir = '/Users/boris/CMBAgents/cmbagent/output/control/data/' # Example path
    train_df_path = output_data_dir + "processed_train_df.pkl"
    test_df_path = output_data_dir + "processed_test_df.pkl"

    # Load the DataFrames
    processed_train_df = pd.read_pickle(train_df_path)
    processed_test_df = pd.read_pickle(test_df_path)

    # Display the first few rows to confirm successful loading
    print("Processed Train DataFrame head:")
    print(processed_train_df.head())
    print("\nProcessed Test DataFrame head:")
    print(processed_test_df.head())

    # Display shapes
    print(f"\nProcessed Train DataFrame shape: {processed_train_df.shape}")
    print(f"Processed Test DataFrame shape: {processed_test_df.shape}")

The `processed_train_df` contains 45,211 samples and 31 columns (30 features + 1 target variable 'y'). The `processed_test_df` contains 4,521 samples and 31 columns.

## Understanding the Features

The features in the processed DataFrames are derived from the original dataset. The preprocessing involved scaling for numeric features, ordinal encoding for features with inherent order, and one-hot encoding for nominal categorical features. Binary features were mapped to 0/1.

### Original Features and Their Context

The original dataset contained 17 features and one target variable.

| # | Feature   | Description                                                                 | Type (Original) | Possible Values (Original Examples)                                                                                                | Banking Domain Context                                                                                                                               |
|---|-----------|-----------------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | age       | Client's age.                                                               | Numeric         | e.g., 18-95                                                                                                                        | Age can influence financial behavior, risk appetite, and product needs. Younger clients might be new to banking, older ones might seek stable investments. |
| 2 | job       | Type of job.                                                                | Categorical     | "admin.", "management", "student", "blue-collar", etc. (12 categories)                                                            | Occupation often correlates with income level, financial stability, and potentially interest in specific banking products.                             |
| 3 | marital   | Marital status.                                                             | Categorical     | "married", "single", "divorced"                                                                                                    | Marital status can affect financial responsibilities and long-term planning.                                                                         |
| 4 | education | Level of education.                                                         | Categorical     | "unknown", "primary", "secondary", "tertiary"                                                                                      | Education level can be an indicator of socio-economic status and financial literacy.                                                                 |
| 5 | default   | Has credit in default?                                                      | Binary          | "yes", "no"                                                                                                                        | Indicates creditworthiness. Clients in default are less likely to be targeted for new investments.                                                   |
| 6 | balance   | Average yearly balance, in euros.                                           | Numeric         | e.g., -8019 to 102127                                                                                                              | Represents the client's current financial holdings with the bank, indicating capacity to invest.                                                     |
| 7 | housing   | Has housing loan?                                                           | Binary          | "yes", "no"                                                                                                                        | Existing housing loans might impact disposable income and willingness to take on new financial products.                                             |
| 8 | loan      | Has personal loan?                                                          | Binary          | "yes", "no"                                                                                                                        | Similar to housing loans, personal loans can affect a client's financial flexibility.                                                                |
| 9 | contact   | Contact communication type for the current campaign.                        | Categorical     | "unknown", "telephone", "cellular"                                                                                                 | Method used to reach the client. Cellular is often more direct and modern. "Unknown" might represent older methods or unrecorded types.            |
| 10| day       | Last contact day of the month.                                              | Numeric         | 1-31                                                                                                                               | Day of contact might have minor effects, possibly related to paydays or general client availability.                                                 |
| 11| month     | Last contact month of year.                                                 | Categorical     | "jan", "feb", ..., "dec"                                                                                                           | Month of contact can reveal seasonal trends in client responsiveness, possibly linked to economic cycles or personal financial planning periods.      |
| 12| duration  | Last contact duration, in seconds.                                          | Numeric         | e.g., 0-4918                                                                                                                       | Duration of the call. Longer calls might indicate higher client engagement or interest. *Crucial note below.*                                       |
| 13| campaign  | Number of contacts performed during this campaign for this client.          | Numeric         | e.g., 1-63                                                                                                                         | Indicates the intensity of marketing efforts for a specific client in the current campaign.                                                          |
| 14| pdays     | Days since client was last contacted from a previous campaign (-1 if never). | Numeric         | e.g., -1 to 871                                                                                                                    | Measures recency of previous interactions. Lower positive values mean more recent contact. -1 indicates no prior contact.                            |
| 15| previous  | Number of contacts performed before this campaign for this client.          | Numeric         | e.g., 0-275                                                                                                                        | Indicates history of interaction with the client in past campaigns.                                                                                  |
| 16| poutcome  | Outcome of the previous marketing campaign.                                 | Categorical     | "unknown", "other", "failure", "success"                                                                                           | Key indicator of past responsiveness to marketing, highly influential for future success.                                                            |
| 17| y         | Has the client subscribed a term deposit? (Target variable)                 | Binary          | "yes", "no"                                                                                                                        | The outcome variable to be predicted.                                                                                                                |

### Processed Features

The preprocessing script (Step 3) transformed these original features as follows:

1.  **Binary Mapping**:
    *   `default`, `housing`, `loan`, and the target `y` were mapped from "yes"/"no" to 1/0.

2.  **ColumnTransformer Operations**:
    *   **Numeric Features**: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`.
        *   Transformation: `StandardScaler` (zero mean, unit variance).
        *   Processed Names: Prefixed with `num__` (e.g., `num__age`, `num__balance`).
    *   **Ordinal Features**: `education`, `month`.
        *   Transformation: `OrdinalEncoder`.
            *   `education` categories: `['unknown', 'primary', 'secondary', 'tertiary']` (encoded as 0-3).
            *   `month` categories: `['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']` (encoded as 0-11).
        *   Processed Names: Prefixed with `ord__` (e.g., `ord__education`, `ord__month`).
    *   **Nominal Features**: `job`, `marital`, `contact`, `poutcome`.
        *   Transformation: `OneHotEncoder` (with `drop='first'` to avoid multicollinearity, `handle_unknown='ignore'`).
        *   Processed Names: Prefixed with `nom__` followed by the original feature name and category (e.g., `nom__job_management`, `nom__marital_single`).
    *   **Passthrough Features**: The manually mapped binary features (`default`, `housing`, `loan`) were passed through the `ColumnTransformer` without further changes due to `remainder='passthrough'`. They retain their original names in the processed DataFrames (e.g., `default`, `housing`, `loan`) and their 0/1 values.

The target variable `y` remains as a single column with 0/1 values.

## Feature Influence on Term Deposit Subscription (y)

The EDA (Step 2) provided insights into how different features relate to the likelihood of a client subscribing to a term deposit.

*   **Client Demographics**:
    *   `age` (`num__age`): EDA showed a slight tendency for older clients to subscribe. The boxplot `age_vs_y_boxplot_3` indicated a slightly higher median age for subscribers.
    *   `job` (e.g., `nom__job_retired`, `nom__job_student`): Certain job categories showed higher subscription rates. Specifically, `job_retired` and `job_student` had positive correlations with 'y' (0.08 and 0.08 respectively). `job_blue-collar` had a negative correlation (-0.07).
    *   `marital` (e.g., `nom__marital_single`): `marital_single` showed a small positive correlation (0.06), while `marital_married` had a negative correlation (-0.06).
    *   `education` (`ord__education`): Higher education levels, particularly `education_tertiary` (represented by higher ordinal values), showed a positive correlation with subscription (0.07).

*   **Client Financial Status**:
    *   `default` (`default`): Having credit in default (`default`=1) showed a negative correlation (-0.02) with subscription, as expected.
    *   `balance` (`num__balance`): Clients with higher balances tended to subscribe more, though the distribution is heavily skewed. The correlation was positive but modest (0.05). `balance_vs_y_boxplot_5` showed a higher median balance for subscribers.
    *   `housing` (`housing`): Having a housing loan (`housing`=1) was negatively correlated (-0.14) with subscription.
    *   `loan` (`loan`): Similarly, having a personal loan (`loan`=1) was negatively correlated (-0.07).

*   **Current Campaign Contact Details**:
    *   `contact` (e.g., `nom__contact_cellular`, `nom__contact_telephone`): The EDA indicated that `contact_unknown` (which includes telephone contacts not explicitly marked as 'telephone' or 'cellular') had a notable negative correlation (-0.15) with subscription. Cellular contact is generally more effective.
    *   `day` (`num__day`): Showed a very weak negative correlation (-0.03). Its impact seems minimal.
    *   `month` (`ord__month`): This feature revealed significant seasonality. Subscription rates were notably higher in March, September, October, and December (positive correlations for these months after one-hot encoding in EDA, e.g., `month_mar` 0.13). Conversely, May, the month with the most contacts, had a strong negative correlation (-0.10). The plot `month_vs_y_rate_14` clearly illustrated this.
    *   `duration` (`num__duration`): **This is a highly influential feature.** The EDA showed a strong positive correlation (0.39) with subscription. The mean duration for successful subscriptions (537 seconds) was much higher than for unsuccessful ones (221 seconds).
        *   **Critical Caveat**: The duration of a call is known only *after* the call has concluded. Therefore, `duration` cannot be used as a predictor if the goal is to identify clients *before* initiating contact. Including it in such a predictive model would lead to an overly optimistic and unrealistic performance. It is useful for benchmarking or analyzing call effectiveness post-campaign but should typically be excluded from pre-call predictive models.

*   **Other Attributes (Previous Campaign History)**:
    *   `campaign` (`num__campaign`): The number of contacts made during the current campaign showed a negative correlation (-0.07). The `campaign_vs_y_rate_10` plot indicated that subscription rates tend to decrease as the number of contacts for the same client in the current campaign increases, especially after the first few attempts.
    *   `pdays` (`num__pdays`): Number of days since the client was last contacted from a previous campaign. A value of -1 (meaning not previously contacted) is common. For clients who *were* previously contacted (pdays > -1), there's a higher propensity to subscribe. The EDA showed a positive correlation (0.10) for `pdays` (when -1 is treated as a large negative number or handled appropriately). The `pdays_status_vs_y_11` plot showed that previously contacted clients had a subscription rate of 23% vs. 9% for those not previously contacted.
    *   `previous` (`num__previous`): Number of contacts performed before this campaign. This feature had a positive correlation (0.09). More previous contacts, especially if successful, can increase subscription likelihood, as seen in `previous_contacts_vs_y_rate_12`.
    *   `poutcome` (e.g., `nom__poutcome_success`): The outcome of the previous marketing campaign is highly predictive. `poutcome_success` had a strong positive correlation (0.31) with current subscription. The `poutcome_vs_y_13` plot showed that 64.7% of clients whose previous outcome was 'success' subscribed in the current campaign. `poutcome_unknown` (the majority category) had a negative correlation (-0.17).

## Recommendations for Feature Selection and Engineering

Based on the EDA and feature characteristics:

*   **Feature Selection**:
    *   **`duration` (`num__duration`)**: As highlighted, this feature should be carefully considered. If the objective is to build a model to predict *which customers to call*, `duration` must be excluded. If the model is for understanding factors contributing to success *after* calls are made, it can be included but its dominance should be noted.
    *   **Low Variance/Correlation Features**: While `StandardScaler` handles zero variance, some one-hot encoded features resulting from rare categories (e.g., `nom__job_unknown` had a near-zero correlation) might offer little predictive power. However, tree-based models can often handle such features effectively. Consider their removal only after initial modeling attempts if they don't contribute.
    *   The `day` feature (`num__day`) showed very weak correlation and might be a candidate for removal if model simplicity is desired.

*   **Feature Engineering (Further Considerations)**:
    The current preprocessing is quite comprehensive. However, further refinements could be explored:
    *   **Interaction Terms**: Explore interactions between highly relevant features, e.g., `poutcome_success` and `num__previous`, or `ord__month` with certain `nom__job` categories.
    *   **`pdays` Handling**: The `-1` value in `pdays` signifies 'not previously contacted'. While `StandardScaler` processes it as a numerical value, creating a binary indicator for 'previously_contacted' (as done in EDA for visualization) or binning `pdays` into categories (e.g., 'not_contacted', 'contacted_recently', 'contacted_long_ago') might capture its impact more effectively for some models. The current scaling treats -1 as just another number, which might not be optimal.
    *   **Non-linear transformations**: For features like `num__balance` or `num__age`, if models struggle with their distributions (despite scaling), applying transformations like log or power transforms (before scaling) or using models inherently good at capturing non-linearities (e.g., tree-based ensembles) is advisable.

## Best Practices for Using the DataFrame in Further Analyses

*   **Model Selection**:
    *   The problem is a binary classification task. Suitable algorithms include Logistic Regression, Support Vector Machines, Decision Trees, Random Forests, Gradient Boosting Machines (like XGBoost, LightGBM, CatBoost), and Neural Networks.
    *   Start with simpler, interpretable models (like Logistic Regression) as a baseline.

*   **Handling Class Imbalance**:
    *   The target variable `y` is imbalanced (approx. 88% 'no', 12% 'yes' in the training set). This must be addressed to prevent models from being biased towards the majority class.
    *   Techniques:
        *   **Resampling**: Oversampling the minority class (e.g., SMOTE - Synthetic Minority Over-sampling Technique), or undersampling the majority class.
        *   **Class Weights**: Many algorithms (e.g., scikit-learn classifiers) support `class_weight='balanced'` or allow manual weight specification.
        *   **Threshold Moving**: Adjusting the decision threshold of the classifier.

*   **Evaluation Metrics**:
    *   Due to class imbalance, **accuracy is not a reliable metric.**
    *   Focus on metrics sensitive to imbalanced data:
        *   **Precision, Recall, F1-score** (especially for the minority class 'yes').
        *   **Area Under the Precision-Recall Curve (AUC-PR)**: Particularly informative for imbalanced datasets.
        *   **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**.
        *   **Confusion Matrix**: To understand the types of errors (false positives, false negatives).

*   **Cross-Validation**:
    *   Use **stratified k-fold cross-validation**. This ensures that each fold maintains approximately the same percentage of samples of each target class as the complete set.

*   **Interpretability**:
    *   If understanding *why* clients subscribe is crucial (not just prediction accuracy), use interpretable models or techniques like SHAP (SHapley Additive exPlanations) values or LIME (Local Interpretable Model-agnostic Explanations) for more complex models.

## Key EDA Findings and Considerations Summary

*   **Significant Class Imbalance**: The target variable 'y' is heavily skewed towards 'no' (non-subscription). This is a primary consideration for modeling.
*   **Strongest Predictive Signals**:
    *   `duration`: Very high correlation, but its use is conditional on the modeling objective (pre-call vs. post-call analysis).
    *   `poutcome_success`: Previous campaign success is a strong indicator of future subscription.
*   **Temporal Effects**: The `month` of contact significantly impacts subscription rates, with peaks observed in spring and autumn/winter months (Mar, Sep, Oct, Dec).
*   **Campaign Fatigue**: The effectiveness of the current campaign (`campaign`) diminishes with an increasing number of contacts to the same client.
*   **Client Profile**:
    *   Clients who were `previously` contacted and had a positive `poutcome` are more likely to subscribe.
    *   Job types like `retired` and `student`, and higher `education` levels (`tertiary`) show a greater propensity to subscribe.
    *   Clients with existing `housing` or personal `loan` commitments are less likely to subscribe.
    *   Clients with credit in `default` are less likely to subscribe.
*   **Contact Method**: `cellular` contact appears more effective than `telephone` or `unknown` contact types.

This detailed notice should facilitate effective use of the preprocessed data for developing predictive models and gaining insights into customer behavior regarding term deposit subscriptions.