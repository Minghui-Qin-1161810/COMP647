import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# Read data
df = pd.read_csv('NZ airfares.csv')

print("=== Basic Data Information ===")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\n=== Data Type Information ===")
print(df.info())

print("\n=== Descriptive Statistics ===")
print(df.describe().transpose())

# Check for duplicate values
print("\n=== Duplicate Value Check ===")
for col in df.columns:
    duplicate_count = df[col].duplicated().sum()
    print(f"Column: {col}")
    print(f"Duplicate count: {duplicate_count}")
    print("*" * 50)

# Check for missing values
print("\n=== Missing Value Analysis ===")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
})
print(missing_info[missing_info['Missing Count'] > 0])

# Show rows with missing values
df_missing_data = df[df.isnull().any(axis=1)]
print(f"\nNumber of rows with missing values: {df_missing_data.shape[0]}")

# Data type classification
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print(f"\nNumerical columns: {list(numerical_columns)}")
print(f"Categorical columns: {list(categorical_columns)}")

# Data cleaning function
def clean_data(df):
    """
    Comprehensive data cleaning function
    """
    df_clean = df.copy()
    
    # 1. Handle missing values
    print("\n=== Handling Missing Values ===")
    
    # Fill numerical columns with median
    for col in numerical_columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"Column '{col}' filled with median {median_val:.2f}")
    
    # Fill categorical columns with mode
    for col in categorical_columns:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"Column '{col}' filled with mode '{mode_val}'")
    
    # 2. Data type conversion
    print("\n=== Data Type Conversion ===")
    
    # Convert date column
    if 'Travel Date' in df_clean.columns:
        df_clean['Travel Date'] = pd.to_datetime(df_clean['Travel Date'], format='%d/%m/%Y', errors='coerce')
        print("Converted 'Travel Date' to datetime type")
    
    # Convert time columns
    time_columns = ['Dep. time', 'Arr. time']
    for col in time_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], format='%I:%M %p', errors='coerce').dt.time
            print(f"Converted '{col}' to time type")
    
    # 3. Handle outliers
    print("\n=== Outlier Detection and Handling ===")
    
    # Use IQR method to detect outliers in numerical columns
    for col in numerical_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"Column '{col}' has {len(outliers)} outliers")
            # Replace outliers with boundary values
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"Outliers clipped to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 4. String cleaning
    print("\n=== String Cleaning ===")
    
    for col in categorical_columns:
        if df_clean[col].dtype == 'object':
            # Remove leading and trailing spaces
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Standardize case
            df_clean[col] = df_clean[col].str.title()
            print(f"Cleaned string format for column '{col}'")
    
    return df_clean

# Execute data cleaning
df_cleaned = clean_data(df)

print("\n=== Cleaned Data Information ===")
print(f"Cleaned dataset shape: {df_cleaned.shape}")
print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")

# Data quality report
print("\n=== Data Quality Report ===")
print(f"Original data rows: {len(df)}")
print(f"Cleaned data rows: {len(df_cleaned)}")
print(f"Data completeness: {((len(df_cleaned) - df_cleaned.isnull().sum().sum()) / (len(df_cleaned) * len(df_cleaned.columns)) * 100):.2f}%")

# Visualization analysis
print("\n=== Generating Data Visualizations ===")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Missing values heatmap
missing_matrix = df.isnull()
sns.heatmap(missing_matrix, cbar=False, ax=axes[0,0])
axes[0,0].set_title('Missing Values Heatmap')
axes[0,0].set_xlabel('Columns')
axes[0,0].set_ylabel('Row Index')

# 2. Distribution of numerical columns
if len(numerical_columns) > 0:
    df_cleaned[numerical_columns].boxplot(ax=axes[0,1])
    axes[0,1].set_title('Numerical Columns Boxplot')
    axes[0,1].set_ylabel('Values')
    axes[0,1].tick_params(axis='x', rotation=45)

# 3. Value counts for categorical columns
if len(categorical_columns) > 0:
    # Select first categorical column for visualization
    cat_col = categorical_columns[0]
    value_counts = df_cleaned[cat_col].value_counts().head(10)
    value_counts.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title(f'{cat_col} Value Distribution')
    axes[1,0].set_xlabel('Categories')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].tick_params(axis='x', rotation=45)

# 4. Correlation heatmap (numerical columns only)
if len(numerical_columns) > 1:
    correlation_matrix = df_cleaned[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Numerical Columns Correlation Heatmap')

plt.tight_layout()
plt.savefig('data_cleaning_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nData cleaning completed! Visualization results saved as 'data_cleaning_analysis.png'")

# Save cleaned data
df_cleaned.to_csv('cleaned_NZ_airfares.csv', index=False)
print("Cleaned data saved as 'cleaned_NZ_airfares.csv'")

# Display cleaned data sample
print("\n=== Cleaned Data Sample ===")
print(df_cleaned.head())
print(f"\nCleaned data types:")
print(df_cleaned.dtypes)