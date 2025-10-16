import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import re
import category_encoders as ce
from collections import Counter
from sklearn.datasets import make_classification

# Set display options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("=" * 80)
print("ASSIGNMENT 2: COMPREHENSIVE DATA ANALYSIS AND PREPROCESSING")
print("=" * 80)

# PART1: DATA CLEANING AND PREPROCESSING


print("\n" + "=" * 60)
print("LAB 2: DATA CLEANING AND PREPROCESSING")
print("=" * 60)

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

# 1. Duplicate handling
# Remove duplicate rows (exact duplicates across all columns)
remove_dumplicates_all_rows = df.drop_duplicates()

# Check for duplicate values in each column
print("\n=== Duplicate Value Check ===")
for col in df.columns:
    duplicate_count = df[col].duplicated().sum()
    print(f"Column: {col}")
    print(f"Duplicate count: {duplicate_count}")
    print("*" * 50)

# 2. Irrelevant data handling
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

# 3. missing data handling
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

# Strategy 4: Fill missing values with statistical measures
# Fill numerical columns with mean
df_filled_mean = df.fillna(df.mean(numeric_only=True))

# Fill numerical columns with median (more robust to outliers)
df_filled_median = df.fillna(df.median(numeric_only=True))

# Fill all columns with mode (most frequent value)
df_filled_mode = df.fillna(df.mode().iloc[0])

# 4. Outlier detection and treatment
# Method 1: Interquartile Range (IQR) Method
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

# Remove outliers using IQR method
df_cleand = df[(df[feature] > lower) & (df[feature] < upper)]

print(f"Cleaned dataset: {df_cleand.shape}")
print(f"Outliers count: {len(df) - len(df_cleand)}")

# Visualize data distribution before and after outlier removal
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
stats.probplot(df[feature], plot=plt)
plt.title("Before Outlier Removal")

plt.subplot(1, 2, 2)
stats.probplot(df_cleand[feature], plot=plt)
plt.title("After Outlier Removal")

plt.suptitle("Outlier Detection and Treatment (IQR Method)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

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
plt.title("Before Z-Score Outlier Removal")

plt.subplot(1, 2, 2)
stats.probplot(df_z_scores_cleaned[feature], plot=plt)
plt.title("After Z-Score Outlier Removal")

plt.suptitle("Outlier Detection and Treatment (Z-Score Method)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 5. data profiling report generation
print("\nGenerating comprehensive data profiling report...")
profile = ProfileReport(df, title="NZ Airfares Data Profiling Report")

# Save report in multiple formats
profile.to_file("ProfilingReport.html")  # Interactive HTML report
profile.to_file("ProfilingReport.json")  # JSON format for programmatic access
print("Data profiling report saved as 'ProfilingReport.html' and 'ProfilingReport.json'")

# PART2: DATA EXPLORATION AND VISUALIZATION

print("\n" + "=" * 60)
print("LAB 3: DATA EXPLORATION AND VISUALIZATION")
print("=" * 60)

# Load the full dataset for exploration
df_ = pd.read_csv('NZ airfares.csv')

# 1. basic data exploratoin
print("=== Basic Data Exploration ===")
print("Dataset head:")
print(df_.head())
print("\nDataset description:")
print(df_.describe().transpose())

# Create a random sample for faster analysis (20% of data)
df = df_.sample(frac=0.20, random_state=42)
print(f"\nUsing 20% random sample for analysis: {df.shape}")
print("Sample description:")
print(df.describe().transpose())

# 2. Data type classfication
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=["object", "category"]).columns

# 3. Date processing and monthly analysis
# Convert Travel Date to datetime format
df['Travel Date'] = pd.to_datetime(df['Travel Date'], format='%d/%m/%Y', errors='coerce')

# Extract month from travel date
df['Month'] = df['Travel Date'].dt.month

# Remove rows with invalid dates
df = df.dropna(subset=['Travel Date'])

# Calculate comprehensive monthly statistics
monthly_stats = df.groupby('Month')['Airfare(NZ$)'].agg(['mean', 'median', 'std', 'count']).round(2)
monthly_stats.columns = ['Mean_Price', 'Median_Price', 'Std_Price', 'Count']
monthly_stats = monthly_stats.reset_index()

# Add month names for better readability
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_stats['Month_Name'] = monthly_stats['Month'].map(lambda x: month_names[x-1])

print("\n=== Monthly Price Statistics ===")
print(monthly_stats)

# 4. Monthly price visualization
plt.figure(figsize=(12, 6))

# Create bar chart for monthly average prices
bars = plt.bar(monthly_stats['Month'], monthly_stats['Mean_Price'], 
               color='skyblue', edgecolor='navy', alpha=0.7)
plt.title('Monthly Average Airfare Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Price (NZ$)', fontsize=12)
plt.xticks(monthly_stats['Month'], monthly_stats['Month_Name'], rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on top of each bar
for i, v in enumerate(monthly_stats['Mean_Price']):
    plt.text(monthly_stats['Month'].iloc[i], v + 5, f'${v:.0f}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 5. Categorical varible analysis
# Select key categorical columns for analysis
selected_categorical_cols = ["Dep. airport", "Arr. airport", "Direct", "Airline"]

# Calculate unique value counts for each categorical column
unique_counts = df[selected_categorical_cols].nunique().sort_values()

print("\n=== Categorical Variable Analysis ===")
print("Unique value counts for categorical columns:")
print(unique_counts)

# Create visualization for categorical variable cardinality
plt.figure(figsize=(12, 6))
sns.barplot(x=unique_counts.index, y=unique_counts.values, palette="Set2")

plt.title("Unique Value Counts for Categorical Columns", fontsize=14, fontweight='bold')
plt.xlabel("Categorical Columns", fontsize=12)
plt.ylabel("Number of Unique Values", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(unique_counts.values):
    plt.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# PART3: FEATURE ENGINEERING AND ENCODING

print("\n" + "=" * 60)
print("LAB 4: FEATURE ENGINEERING AND ENCODING")
print("=" * 60)

# Load the dataset for feature engineering
df = pd.read_csv('NZ airfares.csv')

print("=== Original Dataset Info ===")
print(df.info())
print("\nDataset head:")
print(df.head())

# 1. Data type classification
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# 2. categorical varible distribution analyis
sns.set_theme(style="whitegrid")

# Analyze departure airport distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Dep. airport"], bins=100, kde=True)
plt.title("Departure Airport Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Departure Airport", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze arrival airport distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Arr. airport"], bins=100, kde=True)
plt.title("Arrival Airport Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Arrival Airport", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Feature engineering

# Convert duration string to hours (rounded up)
def convert_duration_to_hours(duration_str):
    """
    Convert duration string (e.g., '1h 25m') to hours (rounded up)
    Returns: float - duration in hours
    """
    if pd.isna(duration_str):
        return np.nan
    
    # Regex pattern to match various duration formats
    pattern = r'(\d+)h\s*(\d+)m|(\d+)h|(\d+)m'
    match = re.search(pattern, str(duration_str))
    
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        
        # Handle case where only minutes are provided
        if match.group(3) is None and match.group(4) is not None:
            hours = 0
            minutes = int(match.group(4))
        
        # Convert to total hours and round up
        total_hours = hours + (minutes / 60)
        return np.ceil(total_hours)
    
    return np.nan

# Apply duration conversion
df['Duration_hours'] = df['Duration'].apply(convert_duration_to_hours)

# Convert travel date to month number
def convert_travel_date_to_month(travel_date_str):
    """
    Convert travel date string to month number (1-12)
    Returns: int - month number
    """
    if pd.isna(travel_date_str):
        return np.nan
    
    try:
        date_obj = pd.to_datetime(travel_date_str, format='%d/%m/%Y', errors='coerce')
        return date_obj.month
    except:
        return np.nan

# Apply date to month conversion
df['Travel_month'] = df['Travel Date'].apply(convert_travel_date_to_month)

# 4. Categorical encoding
# Create duration categories using binning
df["Duration_hours_cat"] = pd.cut(df["Duration_hours"], 
                                  bins=[0, 5, 9, 13, 17, 21, 25, 29], 
                                  labels=["0-4", "4-8", "8-12", "12-16", "16-20", "20-24", "24-28"])

print("\n=== Feature Engineering Results ===")
print("Dataset info after feature engineering:")
print(df.info())
print("\nDataset head after feature engineering:")
print(df.head())

# Label encoding for duration categories
le = LabelEncoder()
df["Duration_hours_cat_lb"] = le.fit_transform(df["Duration_hours_cat"])

# Check for negative duration values (data quality check)
negative_duration = df[df["Duration_hours"] < 0]
print(f"\nNegative duration values found: {len(negative_duration)}")

# Visualize duration category distribution
plt.figure(figsize=(12, 6))
ax = df["Duration_hours_cat"].value_counts().sort_index().plot(kind="bar", color='skyblue', edgecolor='navy')
plt.title("Duration Category Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Duration Categories (Hours)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height(), int(p.get_height()), 
            ha="center", va="bottom", fontweight='bold')

plt.tight_layout()
plt.show()

# 5. ADvanced categorical encoding
# One-hot encoding
one_hot_df = pd.get_dummies(df["Duration_hours_cat"], prefix="Duration_hours_")
df_one_hot = pd.concat([df, one_hot_df], axis=1)
print("\n=== One-Hot Encoding Results ===")
print("Dataset head with one-hot encoding:")
print(df_one_hot.head())

# Binary encoding using category_encoders
encoder = ce.BinaryEncoder(cols=["Duration_hours_cat"])
binary_df = encoder.fit_transform(df["Duration_hours_cat"])
df_binary = pd.concat([df, binary_df], axis=1)
print("\n=== Binary Encoding Results ===")
print("Dataset head with binary encoding:")
print(df_binary.head())

# 6. data scaling
# Select numerical features for scaling
selected_features = ["Airfare(NZ$)", "Duration_hours", "Travel_month"]
airfare_numerical_df = df[selected_features].copy()

print("\n=== Data Scaling ===")
print("Selected features for scaling:")
print(airfare_numerical_df.head())
print("\nOriginal data statistics:")
print(airfare_numerical_df.describe().transpose())

# Min-Max scaling (normalize to [0, 1] range)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_scaled_data = min_max_scaler.fit_transform(airfare_numerical_df)
min_max_scaled_df = pd.DataFrame(min_max_scaled_data, columns=airfare_numerical_df.columns)

print("\nMin-Max scaled data:")
print(min_max_scaled_df.head())
print("Min-Max scaled statistics:")
print(min_max_scaled_df.describe().transpose())

# Standard scaling (z-score normalization)
standard_scaler = StandardScaler()
standard_scaled_data = standard_scaler.fit_transform(airfare_numerical_df)
standard_scaled_df = pd.DataFrame(standard_scaled_data, columns=airfare_numerical_df.columns)

print("\nStandard scaled data:")
print(standard_scaled_df.head())
print("Standard scaled statistics:")
print(standard_scaled_df.describe().transpose())

# 7. Scaling comparison visualization
# Compare scaling effects for Travel_month feature
feature = "Travel_month"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data distribution
sns.histplot(airfare_numerical_df[feature], ax=axes[0], kde=True, bins=5, color='skyblue')
axes[0].set_title("Original Data Distribution", fontweight='bold')
axes[0].set_xlabel(feature)
axes[0].grid(True, alpha=0.3)

# Standard scaled data distribution
sns.histplot(standard_scaled_df[feature], ax=axes[1], kde=True, bins=5, color='lightcoral')
axes[1].set_title("Standard Scaled Distribution", fontweight='bold')
axes[1].set_xlabel(f"{feature} (Standardized)")
axes[1].grid(True, alpha=0.3)

# Min-Max scaled data distribution
sns.histplot(min_max_scaled_df[feature], ax=axes[2], kde=True, bins=5, color='lightgreen')
axes[2].set_title("Min-Max Scaled Distribution", fontweight='bold')
axes[2].set_xlabel(f"{feature} (Min-Max Scaled)")
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Scaling Comparison: {feature}", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Compare scaling effects for Duration_hours feature
feature = "Duration_hours"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data distribution
sns.histplot(airfare_numerical_df[feature], ax=axes[0], kde=True, bins=5, color='skyblue')
axes[0].set_title("Original Data Distribution", fontweight='bold')
axes[0].set_xlabel(feature)
axes[0].grid(True, alpha=0.3)

# Standard scaled data distribution
sns.histplot(standard_scaled_df[feature], ax=axes[1], kde=True, bins=5, color='lightcoral')
axes[1].set_title("Standard Scaled Distribution", fontweight='bold')
axes[1].set_xlabel(f"{feature} (Standardized)")
axes[1].grid(True, alpha=0.3)

# Min-Max scaled data distribution
sns.histplot(min_max_scaled_df[feature], ax=axes[2], kde=True, bins=5, color='lightgreen')
axes[2].set_title("Min-Max Scaled Distribution", fontweight='bold')
axes[2].set_xlabel(f"{feature} (Min-Max Scaled)")
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Scaling Comparison: {feature}", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 8. Imbalanced data handling demostration
print("\n=== Imbalanced Data Handling Demonstration ===")
print("Note: Since airfare prediction is a regression problem, we'll demonstrate imbalanced data concepts on a synthetic classification dataset")

# Create a synthetic imbalanced dataset for demonstration
# 90% class 0 (majority), 10% class 1 (minority)
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                           n_redundant=1, n_classes=2, weights=[0.9, 0.1], random_state=42)

print(f"Original dataset shape: {X.shape}")
print(f"Original class distribution: {np.bincount(y)}")

# Simple oversampling demonstration (without SMOTE)
# Duplicate minority class samples to balance the dataset
minority_indices = np.where(y == 1)[0]
majority_indices = np.where(y == 0)[0]

# Create balanced dataset by duplicating minority samples
n_majority = len(majority_indices)
n_minority = len(minority_indices)
n_duplicates_needed = n_majority - n_minority

# Duplicate minority samples
duplicated_minority_indices = np.random.choice(minority_indices, n_duplicates_needed, replace=True)
all_minority_indices = np.concatenate([minority_indices, duplicated_minority_indices])

# Create balanced dataset
X_balanced = np.vstack([X[majority_indices], X[all_minority_indices]])
y_balanced = np.hstack([y[majority_indices], y[all_minority_indices]])

print(f"Balanced dataset shape: {X_balanced.shape}")
print(f"Balanced class distribution: {np.bincount(y_balanced)}")

# Visualize imbalanced vs balanced data
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original imbalanced data
axes[0].scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0 (Majority)", alpha=0.6, s=30)
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1 (Minority)", alpha=0.8, color="red", s=50)
axes[0].set_title("Original Imbalanced Dataset", fontweight='bold')
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Balanced data
axes[1].scatter(X_balanced[y_balanced == 0, 0], X_balanced[y_balanced == 0, 1], 
                label="Class 0 (Majority)", alpha=0.6, s=30)
axes[1].scatter(X_balanced[y_balanced == 1, 0], X_balanced[y_balanced == 1, 1], 
                label="Class 1 (Minority - Balanced)", alpha=0.8, color="red", s=50)
axes[1].set_title("Balanced Dataset (Simple Oversampling)", fontweight='bold')
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Imbalanced Data Handling: Before and After Balancing", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

"""
REFLECTION ON FEATURE ENGINEERING PROCESS
This dataset contains only one numerical parameter (Airfare), which limits the analytical capabilities, 
I tried to implemented feature engineering techniques to convert non-numerical parameters into numerical parameters. 
I performed the following transformations:

Date Processing: Converted the 'Travel Date' string format into numerical month values (1-12) using pandas datetime conversion.

Duration Conversion: Transformed the 'Duration' string format (e.g., '1h 25m') into numerical hours and then 
categorized them into intervals: '0-4', '4-8', '8-12', '12-16', '16-20', '20-24', '24-28'.

Feature Engineering: Created new numerical features including 'Duration_hours' and 'Travel_month' to enable 
quantitative analysis.

These transformations allowed for more comprehensive statistical analysis, including correlation analysis, 
scaling operations, and machine learning model training on the enhanced dataset.
"""


