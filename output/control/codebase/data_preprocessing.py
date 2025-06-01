# filename: codebase/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import numpy as np

# For SMOTE example, not essential for main script execution if imblearn is not present
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None # Gracefully handle if imblearn is not installed


def preprocess_and_save_data(train_file_path, test_file_path, output_data_dir):
    r"""
    Loads, preprocesses (encodes categorical features, scales numeric features),
    and saves the train and test DataFrames.

    Args:
        train_file_path (str): Path to the training CSV file.
        test_file_path (str): Path to the test CSV file.
        output_data_dir (str): Directory to save the output pickle files.
    """
    print("--- Starting Data Preprocessing ---")

    # --- 1. Load Data ---
    print("Loading data...")
    try:
        train_df = pd.read_csv(train_file_path, sep=';')
        test_df = pd.read_csv(test_file_path, sep=';')
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print("Error: One or both CSV files not found.")
        print(e)
        return
    except Exception as e:
        print("An error occurred while loading the data:")
        print(e)
        return

    # Clean column names (remove quotes)
    train_df.columns = train_df.columns.str.replace('"', '')
    test_df.columns = test_df.columns.str.replace('"', '')
    print("Cleaned column names.")

    # --- 2. Manual Mapping of Binary Features (including target) ---
    binary_cols_map = ['default', 'housing', 'loan', 'y']
    print(r"Manually mapping 'yes'/'no' to 1/0 for columns: " + str(binary_cols_map))
    for df_iter in [train_df, test_df]: # Renamed df to df_iter to avoid conflict
        for col in binary_cols_map:
            if col in df_iter.columns:
                df_iter[col] = df_iter[col].map({'yes': 1, 'no': 0})
            else:
                print(r"Warning: Column '" + col + r"' not found in one of the DataFrames during binary mapping.")
    
    # --- 3. Separate Features (X) and Target (y) ---
    print("Separating features (X) and target (y)...")
    X_train = train_df.drop('y', axis=1)
    y_train = train_df['y']
    X_test = test_df.drop('y', axis=1)
    y_test = test_df['y']

    # --- 4. Define Column Groups and Preprocessor ---
    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    ordinal_features = ['education', 'month']
    education_categories = ['unknown', 'primary', 'secondary', 'tertiary']
    month_categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    nominal_features = ['job', 'marital', 'contact', 'poutcome']

    print("Defining ColumnTransformer for preprocessing...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('ord', OrdinalEncoder(categories=[education_categories, month_categories]), ordinal_features),
            ('nom', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), nominal_features)
        ],
        remainder='passthrough'
    )

    # --- 5. Fit Preprocessor and Transform Data ---
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    print("Transforming training and test data...")
    X_train_processed_np = preprocessor.transform(X_train)
    X_test_processed_np = preprocessor.transform(X_test)
    
    processed_feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(X_train_processed_np, columns=processed_feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=processed_feature_names, index=X_test.index)
    print("Data transformed. Processed X_train shape: " + str(X_train_processed.shape))
    print("Processed X_test shape: " + str(X_test_processed.shape))

    # --- 6. Combine Processed X with y ---
    processed_train_df = pd.concat([X_train_processed, y_train], axis=1)
    processed_test_df = pd.concat([X_test_processed, y_test], axis=1)
    print("Combined processed features with target variable.")

    # --- 7. Save Processed DataFrames ---
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
        print("Created directory: " + output_data_dir)

    train_output_path = os.path.join(output_data_dir, "processed_train_df.pkl")
    test_output_path = os.path.join(output_data_dir, "processed_test_df.pkl")

    try:
        processed_train_df.to_pickle(train_output_path)
        processed_test_df.to_pickle(test_output_path)
        print("\nProcessed train DataFrame saved to: " + train_output_path)
        print("Processed test DataFrame saved to: " + test_output_path)
    except Exception as e:
        print("\nAn error occurred while saving the processed DataFrames:")
        print(e)
        return

    # --- 8. Document Transformations ---
    print("\n--- Summary of Data Transformations ---")
    print("1. Loaded train.csv and test.csv (delimiter ';').")
    print("2. Cleaned column names (removed quotes).")
    print(r"3. Mapped binary features 'default', 'housing', 'loan', and target 'y' ('yes'->1, 'no'->0).")
    print("4. Separated features (X) and target (y).")
    print("5. Applied ColumnTransformer on X (fitted on X_train, applied to X_train and X_test):")
    print(r"   - Numeric features (" + ", ".join(numeric_features) + r"): Scaled using StandardScaler.")
    print(r"   - Ordinal features:")
    print(r"     - 'education': Encoded with order " + str(education_categories) + r".")
    print(r"     - 'month': Encoded with order " + str(month_categories) + r".")
    print(r"   - Nominal features (" + ", ".join(nominal_features) + r"): One-hot encoded (drop='first', handle_unknown='ignore').")
    print(r"   - Manually mapped binary features ('default', 'housing', 'loan') were passed through as is (already 0/1).")
    print("6. Recombined processed X and y into final DataFrames.")
    print(r"7. Saved processed DataFrames as pickle files (" + os.path.basename(train_output_path) + r", " + os.path.basename(test_output_path) + r").")

    # --- 9. Example Code Snippets for Loading ---
    print("\n--- Example Code for Loading Processed Data ---")
    print("import pandas as pd")
    print(r"processed_train_df = pd.read_pickle(r'" + train_output_path + r"')")
    print(r"processed_test_df = pd.read_pickle(r'" + test_output_path + r"')")
    print(r"print('Processed Train DataFrame head:')")
    print(r"print(processed_train_df.head())")
    print(r"print('\nProcessed Test DataFrame head:')")
    print(r"print(processed_test_df.head())")


    # --- 10. Discussion on Class Imbalance ---
    print("\n--- Handling Class Imbalance (Important Note for Modeling) ---")
    y_dist = y_train.value_counts(normalize=True) * 100
    print("The target variable 'y' is imbalanced in the training set:")
    print(y_dist.to_string())
    print("\nThis imbalance should be addressed during the modeling phase. Techniques include:")
    print("  - Resampling techniques:")
    print("    - Oversampling the minority class (e.g., SMOTE - Synthetic Minority Over-sampling Technique).")
    print("    - Undersampling the majority class (e.g., RandomUnderSampler).")
    print("  - Using class weights in model algorithms (e.g., `class_weight='balanced'` in many scikit-learn classifiers).")
    print("  - Choosing appropriate evaluation metrics (e.g., Precision, Recall, F1-score, AUC-PR, ROC AUC instead of just Accuracy).")
    
    if SMOTE is not None:
        print("\nConceptual example of using SMOTE (after loading processed data):")
        print("# Assuming X_train_processed and y_train are available from the loaded pickle file")
        print("# X_train_processed_example = processed_train_df.drop('y', axis=1)")
        print("# y_train_example = processed_train_df['y']")
        print("from imblearn.over_sampling import SMOTE")
        print("smote = SMOTE(random_state=42)")
        print("# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_example, y_train_example)")
        print("# print('Original training dataset target distribution: ')")
        print("# print(y_train_example.value_counts())")
        print("# print('Resampled training dataset target distribution: ')")
        print("# print(pd.Series(y_train_resampled).value_counts())")
        print("# Now use X_train_resampled and y_train_resampled for model training.")
    else:
        print("\n(SMOTE example skipped as 'imblearn' library might not be installed.)")


    print("\n--- Data Preprocessing Finished ---")


if __name__ == "__main__":
    # Define file paths - these would typically be arguments or configured
    default_train_file = "/Users/boris/CMBAgents/demo_data/train.csv"
    default_test_file = "/Users/boris/CMBAgents/demo_data/test.csv"
    
    # Use environment variables if provided, otherwise use defaults
    train_file = os.getenv("TRAIN_FILE_PATH", default_train_file)
    test_file = os.getenv("TEST_FILE_PATH", default_test_file)
    
    # Define output directory for saved data
    output_dir = os.getenv("DATABASE_PATH", "data")
    
    # Ensure the output_dir is an absolute path if it's relative
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    # Check if input files exist
    if not os.path.exists(train_file):
        print("Error: Training file not found at " + train_file)
        exit()
    if not os.path.exists(test_file):
        print("Error: Test file not found at " + test_file)
        exit()

    preprocess_and_save_data(train_file, test_file, output_dir)

    print("\n--- Main script execution complete. ---")
    print("Processed data should be saved in: " + output_dir)