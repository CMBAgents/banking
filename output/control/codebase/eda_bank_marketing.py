# filename: codebase/eda_bank_marketing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

# Matplotlib settings for LaTeX and font
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300  # Default DPI for saved figures

# Global plot counter and timestamp
plot_counter = 1
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Database path
database_path = os.getenv("DATABASE_PATH", "data")
if not os.path.isabs(database_path):
    database_path = os.path.abspath(database_path)  # Ensures it's an absolute path
if not os.path.exists(database_path):
    os.makedirs(database_path)
    print("Created directory: " + database_path)

def save_plot(fig, ax, plot_name_base):
    r"""
    Saves the given matplotlib figure and prints a description.
    Manages a global plot counter and timestamp for unique filenames.

    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        ax (matplotlib.axes.Axes): The main axes object of the plot (used for relim/autoscale).
        plot_name_base (str): Base name for the plot file.
    """
    global plot_counter
    
    # Ensure data limits are updated if necessary
    if ax:  # Check if ax is provided
        ax.relim()
        ax.autoscale_view()
    
    fig.tight_layout()  # Adjust layout to prevent overlapping elements
    
    filename = plot_name_base + "_" + str(plot_counter) + "_" + timestamp + ".png"
    filepath = os.path.join(database_path, filename)
    try:
        fig.savefig(filepath, dpi=300)
        print("Saved plot: " + filepath)
        # Try to get title from the first axes, if available
        plot_title = "Untitled Plot"
        if fig.axes:  # Check if there are any axes
            first_ax_title = fig.axes[0].get_title()
            if first_ax_title:  # Check if title is not empty
                plot_title = first_ax_title
        
        print(r"Description: " + plot_title + r" (Plot " + str(plot_counter) + r")")
        plot_counter += 1
    except Exception as e:
        print("Error saving plot " + filepath + ": " + str(e))
    finally:
        plt.close(fig)  # Close the figure to free memory


def perform_eda(train_file_path):
    r"""
    Performs comprehensive Exploratory Data Analysis (EDA) on the train DataFrame.

    Args:
        train_file_path (str): Path to the training CSV file.
    """
    global plot_counter  # Allow modification of global counter

    # --- 1. Load Data Correctly ---
    print("--- 1. Loading Data ---")
    try:
        train_df = pd.read_csv(train_file_path, sep=';')
        print("Train data loaded successfully.")
        train_df.columns = train_df.columns.str.replace('"', '')  # Clean quotes from column names
    except FileNotFoundError:
        print("Error: " + train_file_path + " not found.")
        return
    except Exception as e:
        print("Error loading " + train_file_path + ": " + str(e))
        return

    print("\n--- 2. Initial Data Inspection ---")
    print("Train DataFrame shape: " + str(train_df.shape))
    print("\nTrain DataFrame head:")
    print(train_df.head())
    print("\nTrain DataFrame info:")
    train_df.info(verbose=True)

    print("\nMissing values per column:")
    print(train_df.isnull().sum())

    print("\n--- 3. Numeric Features Summary ---")
    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    print("Numeric columns: " + str(numeric_cols))
    print(train_df[numeric_cols].describe())

    print("\n--- 4. Categorical Features Summary ---")
    categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
    print("Categorical columns: " + str(categorical_cols))
    print(train_df[categorical_cols].describe())

    for col in categorical_cols:
        print("\nValue counts for " + col + ":")
        print(train_df[col].value_counts())

    print("\n--- 5. Target Variable 'y' Distribution ---")
    print(train_df['y'].value_counts())
    print("\nNormalized distribution:")
    print(train_df['y'].value_counts(normalize=True))

    fig_target, ax_target = plt.subplots(figsize=(6, 4))
    sns.countplot(x='y', data=train_df, ax=ax_target, palette=['#4374B3', '#FF0000'])
    ax_target.set_title(r'Distribution of Target Variable (y)')
    ax_target.set_xlabel(r'Subscribed to Term Deposit?')
    ax_target.set_ylabel(r'Count')
    ax_target.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig_target, ax_target, "target_distribution")

    print("\n--- 6. Visualizing Key Feature Distributions ---")
    key_numeric_features = ['age', 'balance', 'duration']
    for feature in key_numeric_features:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(train_df[feature], kde=True, ax=ax_hist, color='#4374B3')
        ax_hist.set_title(r'Distribution of ' + feature.replace('_', r'\_'))
        ax_hist.set_xlabel(feature.replace('_', r'\_'))
        ax_hist.set_ylabel(r'Frequency')
        ax_hist.grid(True, linestyle='--', alpha=0.7)
        save_plot(fig_hist, ax_hist, feature + "_distribution")

        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='y', y=feature, data=train_df, ax=ax_box, palette=['#4374B3', '#FF0000'])
        ax_box.set_title(r'' + feature.replace('_', r'\_') + r' vs. Target Variable (y)')
        ax_box.set_xlabel(r'Subscribed to Term Deposit?')
        ax_box.set_ylabel(feature.replace('_', r'\_'))
        ax_box.grid(True, linestyle='--', alpha=0.7)
        save_plot(fig_box, ax_box, feature + "_vs_y_boxplot")

    print("\n--- 7. Correlation Analysis ---")
    df_corr = train_df.copy()
    binary_map_cols = ['default', 'housing', 'loan', 'y']
    print("Mapping binary columns to 0/1: " + str(binary_map_cols))
    for col in binary_map_cols:
        df_corr[col] = df_corr[col].map({'yes': 1, 'no': 0})

    categorical_to_encode = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    print("One-hot encoding categorical columns: " + str(categorical_to_encode))
    df_corr = pd.get_dummies(df_corr, columns=categorical_to_encode, drop_first=True)

    print("Shape of DataFrame after encoding: " + str(df_corr.shape))
    correlation_matrix = df_corr.corr()

    fig_corr_heatmap, ax_corr_heatmap = plt.subplots(figsize=(20, 18))  # Increased size
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", ax=ax_corr_heatmap, vmin=-1, vmax=1)
    ax_corr_heatmap.set_title(r'Correlation Matrix of Features (Encoded)')
    save_plot(fig_corr_heatmap, ax_corr_heatmap, "correlation_heatmap_full")

    corr_with_target = correlation_matrix['y'].sort_values(ascending=False)
    print("\nCorrelations with target variable 'y':")
    print(corr_with_target)

    fig_corr_target, ax_corr_target = plt.subplots(figsize=(10, max(8, len(corr_with_target)//3)))  # Dynamic height
    corr_with_target.drop('y').plot(kind='barh', ax=ax_corr_target, color='#4374B3')
    ax_corr_target.set_title(r'Feature Correlation with Target Variable (y)')
    ax_corr_target.set_xlabel(r'Correlation Coefficient')
    ax_corr_target.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig_corr_target, ax_corr_target, "correlation_with_target")

    print("\n--- 8. Relationship between Campaign Features and Target ---")
    print("\nMean duration for each target class:")
    print(train_df.groupby('y')['duration'].mean())

    campaign_cap = train_df['campaign'].quantile(0.99)
    df_campaign_analysis = train_df.copy()
    df_campaign_analysis['campaign_capped'] = df_campaign_analysis['campaign'].clip(upper=campaign_cap)
    campaign_y_rate = df_campaign_analysis.groupby('campaign_capped')['y'].value_counts(normalize=True).mul(100).rename('percentage').unstack(fill_value=0)
    
    if 'yes' in campaign_y_rate.columns:
        fig_camp, ax_camp = plt.subplots(figsize=(12, 6))
        campaign_y_rate['yes'].plot(kind='bar', ax=ax_camp, color='#4374B3')
        ax_camp.set_title(r'Subscription Rate vs. Campaign Contacts (Capped at ' + str(int(campaign_cap)) + r')')
        ax_camp.set_xlabel(r'Number of Campaign Contacts')
        ax_camp.set_ylabel(r'Subscription Rate (\%)')
        ax_camp.grid(True, linestyle='--', alpha=0.7)
        save_plot(fig_camp, ax_camp, "campaign_vs_y_rate")
    else:
        print("No 'yes' outcomes for campaign analysis, skipping plot.")

    pdays_impact = train_df.copy()
    pdays_impact['pdays_contacted_previously'] = pdays_impact['pdays'].apply(lambda x: 'Not Previously Contacted' if x == -1 else 'Previously Contacted')
    print("\nSubscription outcome by 'pdays' status:")
    print(pdays_impact.groupby('pdays_contacted_previously')['y'].value_counts(normalize=True).mul(100))
    
    fig_pdays, ax_pdays = plt.subplots(figsize=(8, 5))
    sns.countplot(x='pdays_contacted_previously', hue='y', data=pdays_impact, ax=ax_pdays, palette=['#4374B3', '#FF0000'])
    ax_pdays.set_title(r'Subscription Outcome by Previous Contact Status (pdays)')
    ax_pdays.set_xlabel(r'Previous Contact Status')
    ax_pdays.set_ylabel(r'Count')
    ax_pdays.legend(title=r'Subscribed')
    ax_pdays.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig_pdays, ax_pdays, "pdays_status_vs_y")

    previous_cap = train_df['previous'].quantile(0.99)
    df_previous_analysis = train_df.copy()
    df_previous_analysis['previous_capped'] = df_previous_analysis['previous'].clip(upper=previous_cap)
    previous_y_rate = df_previous_analysis.groupby('previous_capped')['y'].value_counts(normalize=True).mul(100).rename('percentage').unstack(fill_value=0)

    if 'yes' in previous_y_rate.columns:
        fig_prev, ax_prev = plt.subplots(figsize=(12, 6))
        previous_y_rate['yes'].plot(kind='bar', ax=ax_prev, color='#4374B3')
        ax_prev.set_title(r'Subscription Rate vs. Previous Contacts (Capped at ' + str(int(previous_cap)) + r')')
        ax_prev.set_xlabel(r'Number of Previous Contacts')
        ax_prev.set_ylabel(r'Subscription Rate (\%)')
        ax_prev.grid(True, linestyle='--', alpha=0.7)
        save_plot(fig_prev, ax_prev, "previous_contacts_vs_y_rate")
    else:
        print("No 'yes' outcomes for previous contacts analysis, skipping plot.")

    print("\nAnalysis of 'poutcome':")
    poutcome_y_norm = pd.crosstab(train_df['poutcome'], train_df['y'], normalize='index').mul(100)
    print(poutcome_y_norm)

    fig_pout, ax_pout = plt.subplots(figsize=(10, 6))
    poutcome_y_norm.plot(kind='bar', stacked=False, ax=ax_pout, color=['#4374B3', '#FF0000'])
    ax_pout.set_title(r'Subscription Rate by Previous Campaign Outcome (poutcome)')
    ax_pout.set_xlabel(r'Previous Campaign Outcome')
    ax_pout.set_ylabel(r'Percentage (\%)')
    ax_pout.legend(title=r'Subscribed')
    ax_pout.grid(True, axis='y', linestyle='--', alpha=0.7)
    save_plot(fig_pout, ax_pout, "poutcome_vs_y")

    print("\n--- 9. Temporal Patterns (Month) ---")
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    # Ensure 'month' column exists and is of object type before converting to categorical
    if 'month' in train_df.columns and train_df['month'].dtype == 'object':
        train_df['month'] = pd.Categorical(train_df['month'], categories=month_order, ordered=True)
        
        month_y_rate = train_df.groupby('month')['y'].value_counts(normalize=True).mul(100).rename('percentage').unstack(fill_value=0)
        
        if 'yes' in month_y_rate.columns:
            fig_month, ax_month = plt.subplots(figsize=(12, 6))
            month_y_rate['yes'].plot(kind='bar', ax=ax_month, color='#4374B3')
            ax_month.set_title(r'Subscription Rate by Month')
            ax_month.set_xlabel(r'Month')
            ax_month.set_ylabel(r'Subscription Rate (\%)')
            ax_month.tick_params(axis='x', rotation=45)
            ax_month.grid(True, axis='y', linestyle='--', alpha=0.7)
            save_plot(fig_month, ax_month, "month_vs_y_rate")
        else:
            print("No 'yes' outcomes for monthly analysis, skipping plot.")
    else:
        print("Skipping monthly analysis: 'month' column not found or not of expected type.")

    print("\n--- 10. Documentation of Transformations for EDA ---")
    print("EDA transformations performed on a copy of the DataFrame for correlation analysis:")
    print("- Mapped 'yes'/'no' to 1/0 for: 'default', 'housing', 'loan', 'y'.")
    print("- One-hot encoded: 'job', 'marital', 'education', 'contact', 'month', 'poutcome' (using drop_first=True).")
    print("For specific analyses (campaign, previous contacts), features were capped at their 99th percentile to handle outliers in visualizations.")
    print("Month feature was ordered chronologically for temporal analysis.")

    print("\n--- EDA Finished ---")


if __name__ == "__main__":
    # This path should be set by the agent's environment or configuration.
    # Defaulting to the path from the problem description for this specific task.
    train_csv_path = "/Users/boris/CMBAgents/demo_data/train.csv"
    
    if not os.path.exists(train_csv_path):
        print("Error: Train CSV file not found at " + train_csv_path)
        print("Please ensure the file path is correct.")
    else:
        print("Starting EDA process for: " + train_csv_path)
        perform_eda(train_csv_path)
        print("EDA script execution complete.")