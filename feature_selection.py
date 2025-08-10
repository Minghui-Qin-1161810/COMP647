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

# Load cleaned data
print("=== Loading Cleaned Data ===")
df = pd.read_csv('cleaned_NZ_airfares.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert Travel Date to datetime if it's not already
if 'Travel Date' in df.columns:
    df['Travel Date'] = pd.to_datetime(df['Travel Date'])

print("\n=== Feature Engineering ===")

# 1. Extract date features
print("1. Extracting date features...")
if 'Travel Date' in df.columns:
    df['Year'] = df['Travel Date'].dt.year
    df['Month'] = df['Travel Date'].dt.month
    df['Day'] = df['Travel Date'].dt.day
    df['DayOfWeek'] = df['Travel Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsHoliday'] = df['Month'].isin([12, 1, 2]).astype(int)  # Summer holidays
    print("   - Added: Year, Month, Day, DayOfWeek, IsWeekend, IsHoliday")

# 2. Extract time features
print("2. Extracting time features...")
if 'Dep. time' in df.columns:
    # Convert time to hour
    df['DepHour'] = pd.to_datetime(df['Dep. time'], format='%H:%M:%S').dt.hour
    df['IsPeakHour'] = df['DepHour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['IsEarlyMorning'] = (df['DepHour'] < 6).astype(int)
    df['IsLateNight'] = (df['DepHour'] > 22).astype(int)
    print("   - Added: DepHour, IsPeakHour, IsEarlyMorning, IsLateNight")

# 3. Extract duration features
print("3. Extracting duration features...")
if 'Duration' in df.columns:
    # Extract hours and minutes from duration string
    df['Duration_Hours'] = df['Duration'].str.extract(r'(\d+)h').astype(float)
    df['Duration_Minutes'] = df['Duration'].str.extract(r'(\d+)m').astype(float)
    df['Duration_Total_Minutes'] = df['Duration_Hours'] * 60 + df['Duration_Minutes']
    df['IsShortFlight'] = (df['Duration_Total_Minutes'] < 60).astype(int)
    df['IsLongFlight'] = (df['Duration_Total_Minutes'] > 120).astype(int)
    print("   - Added: Duration_Hours, Duration_Minutes, Duration_Total_Minutes, IsShortFlight, IsLongFlight")

# 4. Extract route features
print("4. Extracting route features...")
if 'Dep. airport' in df.columns and 'Arr. airport' in df.columns:
    df['Route'] = df['Dep. airport'] + '-' + df['Arr. airport']
    df['IsPopularRoute'] = df['Route'].isin(df['Route'].value_counts().head(10).index).astype(int)
    print("   - Added: Route, IsPopularRoute")

# 5. Extract airline features
print("5. Extracting airline features...")
if 'Airline' in df.columns:
    df['IsMajorAirline'] = df['Airline'].isin(['Air New Zealand', 'Jetstar']).astype(int)
    print("   - Added: IsMajorAirline")

# 6. Extract direct flight features
print("6. Extracting direct flight features...")
if 'Direct' in df.columns:
    df['IsDirect'] = (df['Direct'] == '(Direct)').astype(int)
    print("   - Added: IsDirect")

# 7. Extract baggage features
print("7. Extracting baggage features...")
if 'Baggage' in df.columns:
    df['HasBaggage'] = (df['Baggage'] != 'Checked Bag Not Included').astype(int)
    print("   - Added: HasBaggage")

# Clean the data - handle NaN and infinite values more carefully
print("\n=== Data Cleaning ===")
print("Handling NaN and infinite values...")

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Fill NaN values in numerical columns with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print(f"Dataset shape after cleaning: {df.shape}")

print("\n=== Feature Selection ===")

# Separate numerical and categorical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Remove target variable from features
target = 'Airfare(NZ$)'
if target in numerical_features:
    numerical_features.remove(target)

print(f"\nTarget variable: {target}")
print(f"Features for selection: {len(numerical_features)}")

# 1. Correlation-based feature selection
print("\n1. Correlation-based feature selection...")
correlation_matrix = df[numerical_features + [target]].corr()
target_correlations = correlation_matrix[target].abs().sort_values(ascending=False)
print("Top 10 features by correlation with target:")
print(target_correlations.head(10))

# Select features with correlation > 0.05
high_corr_features = target_correlations[target_correlations > 0.05].index.tolist()
if target in high_corr_features:
    high_corr_features.remove(target)
print(f"Features with correlation > 0.05: {len(high_corr_features)}")

# 2. Variance-based feature selection
print("\n2. Variance-based feature selection...")
feature_variance = df[numerical_features].var().sort_values(ascending=False)
print("Top 10 features by variance:")
print(feature_variance.head(10))

# Select features with variance > 0.001
high_var_features = feature_variance[feature_variance > 0.001].index.tolist()
print(f"Features with variance > 0.001: {len(high_var_features)}")

# 3. Statistical significance test
print("\n3. Statistical significance test...")
significant_features = []
for feature in numerical_features:
    if feature != target:
        try:
            # Check if feature has variance
            if df[feature].var() > 0:
                correlation, p_value = stats.pearsonr(df[feature], df[target])
                if p_value < 0.05:  # 95% confidence level
                    significant_features.append(feature)
        except:
            continue
print(f"Statistically significant features (p < 0.05): {len(significant_features)}")

# 4. Final feature selection
print("\n4. Final feature selection...")
# Use union of different selection methods
final_features = list(set(high_corr_features) | set(high_var_features) | set(significant_features))

print(f"Final selected features: {len(final_features)}")
print("Selected features:")
for i, feature in enumerate(final_features, 1):
    print(f"  {i}. {feature}")

# Create feature importance visualization
print("\n=== Feature Importance Visualization ===")

# Create correlation heatmap
plt.figure(figsize=(12, 8))
selected_features = final_features + [target]
correlation_subset = df[selected_features].corr()
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create feature importance bar plot
plt.figure(figsize=(10, 6))
feature_importance = target_correlations[final_features].sort_values(ascending=True)
plt.barh(range(len(feature_importance)), feature_importance.values)
plt.yticks(range(len(feature_importance)), feature_importance.index)
plt.xlabel('Correlation with Target')
plt.title('Feature Importance (Correlation with Airfare)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Save selected features dataset
print("\n=== Saving Results ===")
selected_df = df[final_features + [target]]
selected_df.to_csv('selected_features_dataset.csv', index=False)
print(f"Selected features dataset saved: {selected_df.shape}")

# Print summary
print("\n=== Feature Selection Summary ===")
print(f"Original features: {len(df.columns) - 1}")  # Exclude target
print(f"Selected features: {len(final_features)}")
print(f"Reduction: {((len(df.columns) - 1 - len(final_features)) / (len(df.columns) - 1) * 100):.1f}%")

print("\nSelected features for modeling:")
for feature in final_features:
    corr = target_correlations[feature]
    print(f"  - {feature}: correlation = {corr:.3f}")

print("\nFeature selection completed!")
print("Files generated:")
print("  - selected_features_dataset.csv: Dataset with selected features")
print("  - feature_correlation_heatmap.png: Correlation visualization")
print("  - feature_importance.png: Feature importance plot")
