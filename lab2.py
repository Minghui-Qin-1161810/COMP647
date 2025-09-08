import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ydata_profiling import ProfileReport

# Set display options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the dataset
df = pd.read_csv('NZ airfares.csv')

print("=== Basic Data Information ===")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\n=== Data Information ===")
print(df.info())

print("\n=== Descriptive Statistics ===")
print(df.describe().transpose())

# 1. DUPLICATE HANDLING
# Remove duplicate rows (exact duplicates across all columns)
remove_dumplicates_all_rows = df.drop_duplicates()

# Check for duplicate values in each column
print("\n=== Duplicate Value Check ===")
for col in df.columns:
    duplicate_count = df[col].duplicated().sum()
    print(f"Column: {col}")
    print(f"Duplicate count: {duplicate_count}")
    print("*" * 50)

# 2. IRRELEVANT DATA HANDLING
# Identify and remove constant features (columns with only one unique value)
constant_features = [col for col in df.columns if df[col].nunique() == 1]
print("Constant featues: ", constant_features)

# Remove constant features from the dataframe
df_no_constant_features = df.drop(columns=constant_features)

# Identify columns with high missing value ratio
threshold = 5
print(f"Total records {df.shape[0]}")
print("*" * 50)
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_ratio = (missing_count / df.shape[0]) * 100
    if missing_ratio > threshold:
        print(f"Colmn: {col} has {missing_count} missing values ({missing_ratio:.2f})")
        print("*" * 50)

# Remove columns with more than 5% missing values
columns_to_drop = [col for col in df_no_constant_features.columns if (df_no_constant_features[col].isnull().sum() / df_no_constant_features.shape[0]) * 100 > threshold]
df_low_missing_data = df_no_constant_features.drop(columns=columns_to_drop)

# 3. MISSING VALUE HANDLING
# Show rows with missing values
df_missing_data = df[df.isnull().any(axis=1)]
print(f"\nNumber of rows with missing values: {df_missing_data.shape[0]}")
df_missing_data.tail()

# Data type classification for targeted missing value treatment
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print(f"Numerical columns: ", numerical_columns)
print(f"Categorical columns: ", categorical_columns)

# Identify columns with missing values by data type (numerical)
missing_numerical_columns = df[numerical_columns].isnull().any()
missing_numerical_columns = missing_numerical_columns[missing_numerical_columns].index
print("Numerical columns with missing values:", missing_numerical_columns.to_list())

# Identify columns with missing values by data type (categorical)
missing_categorical_columns = df[categorical_columns].isnull().any()
missing_categorical_columns = missing_categorical_columns[missing_categorical_columns].index
print("Categorical columns with missing values:", missing_categorical_columns.to_list())

# Strategy 1: Remove rows with any missing values
df_no_missing_data = df.dropna()

# Strategy 2: Remove columns with any missing values
df_no_missing_columns = df.dropna(axis=1)

# Strategy 3: Fill missing values with specific values
df_filled = df.copy()
df_filled[numerical_columns] = df_filled[numerical_columns].fillna(0)
for col in categorical_columns:
    df_filled[col] = df_filled[col].fillna("Unknown")

# Check the filled data in dataframe
# selected_rows = df_filled.iloc(value)
# selected_rows

# Strategy 4: Fill missing values with statistical measures

# Fill numerical columns with mean
df_filled_mean = df.fillna(df.mean(numeric_only=True))

# Fill numerical columns with median (more robust to outliers)
df_filled_median = df.fillna(df.median(numeric_only=True))

# Fill all columns with mode (most frequent value)
# .iloc[0] gets the first mode if multiple modes exist
df_filled_mode = df.fillna(df.mode().iloc[0])

# Check the filled values by mean
# df_filled_mean_rows = df_filled_mean.iloc[[value]]
# df_filled_mean_rows

# Check the filled values by median
# df_filled_median_rows = df_filled_median.iloc[[value]]
# df_filled_median_rows

# Check the filled values by mode
# df_filled_mode_rows = df_filled_mode.iloc[[value]]
# df_filled_mode_rows

# 4. OUTLIER DETECTION AND TREATMENT
# Method 1: Interquartile Range (IQR) Method
# Outliers are points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
def find_outliers_IQR(input_df, variable):
    """
    Find outlier limits using IQR method
    Returns lower and upper limits for outlier detection
    """
    Q1 = input_df[variable].quantile(0.25)
    Q3 = input_df[variable].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - (IQR * 1.5)
    upper_limit = Q3 + (IQR * 1.5)

    return lower_limit, upper_limit

# Apply IQR method to target variable (Airfare)
feature = "Airfare(NZ$)"
lower, upper = find_outliers_IQR(df, feature)
# lower, upper
# print(lower)
# print(upper)

# Remove outliers using IQR method
df_cleand = df[(df[feature] > lower) & (df[feature] < upper)]

print(f"Cleaned dataset: {df_cleand.shape}")
print(f"Outliers count: {len(df) - len(df_cleand)}")

# Visualize data distribution before and after outlier removal
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
stats.probplot(df[feature], plot=plt)

plt.subplot(1, 2, 2)
stats.probplot(df_cleand[feature], plot=plt)

def find_outliers_ZScore(input_df, variable):
    """
    Calculate Z-scores for outlier detection
    Returns dataframe with Z-scores added as new column
    """
    df_z_scores = input_df.copy()

    # Calculate Z-scores for the specified variable (excluding NaN values)
    z_scores = np.abs(stats.zscore(input_df[variable].dropna()))

    # Add Z-scores as a new column
    df_z_scores[variable + "_Z"] = z_scores
    return df_z_scores

# Apply Z-Score method to target variable
df_z_scores = find_outliers_ZScore(df.copy(), feature)
df_z_scores.head()

# Remove outliers where |Z-score| > 3
df_z_scores_cleaned = df_z_scores[df_z_scores[feature + "_Z"] < 3]

print(f"Cleaned dataset : {df_z_scores_cleaned.shape}")
print(f"Outliers count: {len(df_z_scores) - len(df_z_scores_cleaned)}")

sns.set_style("whitegrid")
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
stats.probplot(df_z_scores[feature], plot=plt)

plt.subplot(1, 2, 2)
stats.probplot(df_z_scores_cleaned[feature], plot=plt)

# 5. DATA PROFILING REPORT GENERATION
# Generate automated data profiling report using ydata-profiling
profile = ProfileReport(df, title="NZ Airfares Data Profiling Report")

# Save report in multiple formats
profile.to_file("ProfilingReport.html")  # Interactive HTML report
profile.to_file("ProfilingReport.json")  # JSON format for programmatic access
