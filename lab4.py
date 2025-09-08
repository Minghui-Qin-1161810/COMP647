import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import re
import category_encoders as ce
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Set display options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the dataset
df = pd.read_csv('NZ airfares.csv')
df.describe().transpose()
df.head()

# 1. DATA TYPE CLASSIFICATION
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# 2. CATEGORICAL VARIABLE DISTRIBUTION ANALYSIS
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

# 3. FEATURE ENGINEERING

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

# 4. CATEGORICAL ENCODING
# Create duration categories using binning
df["Duration_hours_cat"] = pd.cut(df["Duration_hours"], 
                                  bins=[0, 5, 9, 13, 17, 21, 25, 29], 
                                  labels=["0-4", "4-8", "8-12", "12-16", "16-20", "20-24", "24-28"])

# Display dataset info after feature engineering
df.info()
df.head()

# Label encoding for duration categories
le = LabelEncoder()
df["Duration_hours_cat_lb"] = le.fit_transform(df["Duration_hours_cat"])

# Check for negative duration values (data quality check)
negative_duration = df[df["Duration_hours"] < 0]

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

# 5. ADVANCED CATEGORICAL ENCODING
# One-hot encoding
one_hot_df = pd.get_dummies(df["Duration_hours_cat"], prefix="Duration_hours_")
df_one_hot = pd.concat([df, one_hot_df], axis=1)
df_one_hot.head()

# Binary encoding using category_encoders
encoder = ce.BinaryEncoder(cols=["Duration_hours_cat"])
binary_df = encoder.fit_transform(df["Duration_hours_cat"])
df_binary = pd.concat([df, binary_df], axis=1)
df_binary.head()

# 6. DATA SCALING
# Select numerical features for scaling
selected_features = ["Airfare(NZ$)", "Duration_hours", "Travel_month"]
airfare_numerical_df = df[selected_features].copy()
airfare_numerical_df.head()
airfare_numerical_df.describe().transpose()

# Min-Max scaling (normalize to [0, 1] range)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_scaled_data = min_max_scaler.fit_transform(airfare_numerical_df)
min_max_scaled_df = pd.DataFrame(min_max_scaled_data, columns=airfare_numerical_df.columns)

min_max_scaled_df.head()
min_max_scaled_df.describe().transpose()

# Standard scaling (z-score normalization)
standard_scaler = StandardScaler()
standard_scaled_data = standard_scaler.fit_transform(airfare_numerical_df)
standard_scaled_df = pd.DataFrame(standard_scaled_data, columns=airfare_numerical_df.columns)

standard_scaled_df.head()
standard_scaled_df.describe().transpose()

# 7. SCALING COMPARISON VISUALIZATION
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

# 8. IMBALANCED DATA HANDLING WITH SMOTE
# Note: Since airfare prediction is a regression problem, we'll demonstrate SMOTE on a synthetic classification dataset

# Create a synthetic imbalanced dataset for demonstration
# 90% class 0 (majority), 10% class 1 (minority)
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                           n_redundant=1, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Apply SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Visualize SMOTE effect
# Calculate synthetic sample statistics
n_minority_before = np.sum(y == 1)
n_minority_after = np.sum(y_resampled == 1)
n_synthetic = n_minority_after - n_minority_before

# Identify synthetic samples (newly generated minority points)
synthetic_points = X_resampled[-(n_synthetic):]

# Create before/after SMOTE visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original imbalanced data
axes[0].scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0 (Majority)", alpha=0.6, s=30)
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1 (Minority)", alpha=0.8, color="red", s=50)
axes[0].set_title("Before SMOTE (Imbalanced Dataset)", fontweight='bold')
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Resampled data with synthetic samples highlighted
axes[1].scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1],
                 label="Class 0 (Majority)", alpha=0.6, s=30)
axes[1].scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1],
                 label="Class 1 (Minority - Original)", alpha=0.8, color="red", s=50)
axes[1].scatter(synthetic_points[:, 0], synthetic_points[:, 1],
                 label="Class 1 (Minority - Synthetic)", alpha=0.9,
                 color="yellow", edgecolor="black", marker="x", s=80)
axes[1].set_title("After SMOTE (Balanced Dataset)", fontweight='bold')
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("SMOTE: Synthetic Minority Oversampling Technique", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


