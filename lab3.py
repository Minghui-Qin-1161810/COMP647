# Lab 3: Data Exploration and Visualization
# This script performs comprehensive data exploration including:
# - Basic data statistics and sampling
# - Monthly price analysis and visualization
# - Categorical variable analysis
# - Data distribution exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set display options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the full dataset
df_ = pd.read_csv('NZ airfares.csv')

# 1. BASIC DATA EXPLORATION
df_.head()
df_.describe().transpose()

# Create a random sample for faster analysis (20% of data)
# Using random_state=42 for reproducible results
df = df_.sample(frac=0.20, random_state=42)
df.describe().transpose()

# 2. DATA TYPE CLASSIFICATION
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=["object", "category"]).columns

# 3. DATE PROCESSING AND MONTHLY ANALYSIS
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

# 4. MONTHLY PRICE VISUALIZATION
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

# 5. CATEGORICAL VARIABLE ANALYSIS
# Select key categorical columns for analysis
selected_categorical_cols = ["Dep. airport", "Arr. airport", "Direct", "Airline"]

# Calculate unique value counts for each categorical column
unique_counts = df[selected_categorical_cols].nunique().sort_values()

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


