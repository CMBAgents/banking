# filename: codebase/data_loading_checks.py
import pandas as pd
import os


def load_and_check_data(train_file_path, test_file_path, output_data_dir):
    r"""
    Loads train and test CSV files into pandas DataFrames, performs basic consistency checks,
    and saves the DataFrames as pickle files.

    Args:
        train_file_path (str): Path to the training CSV file.
        test_file_path (str): Path to the test CSV file.
        output_data_dir (str): Directory to save the output pickle files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
        print("Created directory: " + output_data_dir)

    # --- 1. Read the train and test CSV files ---
    print("Loading data...")
    try:
        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print("Error: One or both CSV files not found.")
        print(e)
        return None, None
    except Exception as e:
        print("An error occurred while loading the data:")
        print(e)
        return None, None

    # --- 2. Display the first few rows of each DataFrame ---
    print("\n--- Train DataFrame Head ---")
    print(train_df.head())
    print("\n--- Test DataFrame Head ---")
    print(test_df.head())

    # --- 3. Print the shape and column names of both DataFrames ---
    print("\n--- Train DataFrame Shape ---")
    print(train_df.shape)
    print("\n--- Test DataFrame Shape ---")
    print(test_df.shape)

    print("\n--- Train DataFrame Columns ---")
    print(train_df.columns.tolist())
    print("\n--- Test DataFrame Columns ---")
    print(test_df.columns.tolist())

    # --- 4. Check for data consistency ---
    print("\n--- Data Consistency Checks ---")

    # Compare column names
    train_cols = train_df.columns.tolist()
    test_cols = test_df.columns.tolist()
    if train_cols == test_cols:
        print("\nColumn names are consistent between train and test sets.")
    else:
        print("\nWARNING: Column names are NOT consistent!")
        print("Train columns: " + str(train_cols))
        print("Test columns: " + str(test_cols))
        # Find differing columns
        missing_in_test = [col for col in train_cols if col not in test_cols]
        missing_in_train = [col for col in test_cols if col not in train_cols]
        if missing_in_test:
            print("Columns in train but not in test: " + str(missing_in_test))
        if missing_in_train:
            print("Columns in test but not in train: " + str(missing_in_train))
        
    # Compare data types
    if train_df.dtypes.equals(test_df.dtypes):
        print("\nData types are consistent across all columns.")
    else:
        print("\nWARNING: Data types are NOT consistent across all columns.")
        for col in train_cols:
            if col in test_cols:  # Check if column exists in both before comparing dtypes
                if train_df[col].dtype != test_df[col].dtype:
                    print("Column '" + col + "': Train dtype = " + str(train_df[col].dtype) + ", Test dtype = " + str(test_df[col].dtype))
            else:
                print("Column '" + col + "' not found in test_df for dtype comparison.")
        # Also check for columns only in test_df
        for col in test_cols:
            if col not in train_cols:
                 print("Column '" + col + "' not found in train_df for dtype comparison.")


    # Compare value ranges for numeric features and unique values for categorical features
    print("\n--- Feature Value Range and Unique Values Comparison ---")
    if train_cols == test_cols:  # Proceed only if columns are the same for meaningful comparison
        for col in train_cols:
            print("\nComparing column: " + col)
            if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(test_df[col]):
                print("  Type: Numeric")
                print("  Train - Min: " + str(train_df[col].min()) + ", Max: " + str(train_df[col].max()) + 
                      ", Mean: " + str(round(train_df[col].mean(), 2)) + ", Std: " + str(round(train_df[col].std(), 2)))
                print("  Test  - Min: " + str(test_df[col].min()) + ", Max: " + str(test_df[col].max()) + 
                      ", Mean: " + str(round(test_df[col].mean(), 2)) + ", Std: " + str(round(test_df[col].std(), 2)))
            elif train_df[col].dtype == 'object' and test_df[col].dtype == 'object':
                print("  Type: Categorical/Object")
                train_unique = set(train_df[col].unique())
                test_unique = set(test_df[col].unique())
                print("  Train Unique Values (" + str(len(train_unique)) + "):", str(sorted(list(train_unique))[:10]) + ("..." if len(train_unique) > 10 else ""))
                print("  Test Unique Values  (" + str(len(test_unique)) + "):", str(sorted(list(test_unique))[:10]) + ("..." if len(test_unique) > 10 else ""))
                
                if train_unique == test_unique:
                    print("  Unique value sets are identical.")
                else:
                    print("  WARNING: Unique value sets differ.")
                    diff_train_test = train_unique - test_unique
                    diff_test_train = test_unique - train_unique
                    if diff_train_test:
                        print("    Values in Train but not in Test: " + str(sorted(list(diff_train_test))[:5]) + ("..." if len(diff_train_test) > 5 else ""))
                    if diff_test_train:
                        print("    Values in Test but not in Train: " + str(sorted(list(diff_test_train))[:5]) + ("..." if len(diff_test_train) > 5 else ""))
            else:
                # This case handles if dtypes were different for the same column name
                print("  Skipping value comparison for column '" + col + "' due to differing or non-standard types.")
                print("  Train dtype: " + str(train_df[col].dtype) + ", Test dtype: " + str(test_df[col].dtype))


    # --- 5. Save the loaded DataFrames for further analysis ---
    train_output_path = os.path.join(output_data_dir, "train_df.pkl")
    test_output_path = os.path.join(output_data_dir, "test_df.pkl")

    try:
        train_df.to_pickle(train_output_path)
        test_df.to_pickle(test_output_path)
        print("\nTrain DataFrame saved to: " + train_output_path)
        print("Test DataFrame saved to: " + test_output_path)
    except Exception as e:
        print("\nAn error occurred while saving the DataFrames:")
        print(e)
        
    return train_df, test_df


if __name__ == "__main__":
    # Define file paths - replace with your actual paths
    # These paths are placeholders and will be replaced by the agent environment
    train_file_path = "/Users/boris/CMBAgents/demo_data/train.csv"
    test_file_path = "/Users/boris/CMBAgents/demo_data/test.csv"
    
    # Define output directory for saved data
    # The agent is instructed to save files under 'data/'
    database_path = os.getenv("DATABASE_PATH", "data")  # Use environment variable if available, else default to 'data'
    
    # Ensure the database_path is an absolute path if it's a relative one like 'data'
    if not os.path.isabs(database_path):
        database_path = os.path.join(os.getcwd(), database_path)

    # Call the function
    train_df, test_df = load_and_check_data(train_file_path, test_file_path, database_path)

    if train_df is not None and test_df is not None:
        print("\n--- Script Finished ---")
        print("DataFrames are loaded and initial checks are complete.")
        print("Saved DataFrames can be found in the '" + database_path + "' directory.")
    else:
        print("\n--- Script Finished with Errors ---")
        print("Data loading or processing failed. Please check the error messages above.")
