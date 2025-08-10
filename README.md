# Flight Price Prediction - Data Cleaning Project

## Project Overview

This project focuses on cleaning and preprocessing New Zealand airfare data for machine learning analysis. The dataset contains flight information including travel dates, airports, times, airlines, and prices.

## Dataset Information

### Original Data
- **File**: `NZ airfares.csv`
- **Size**: 13MB
- **Records**: 162,833 flights
- **Columns**: 11 features

### Data Schema
| Column | Type | Description | Missing Values |
|--------|------|-------------|----------------|
| Travel Date | datetime | Date of travel | 0 |
| Dep. airport | string | Departure airport code | 24 (0.01%) |
| Dep. time | time | Departure time | 0 |
| Arr. airport | string | Arrival airport code | 24 (0.01%) |
| Arr. time | time | Arrival time | 5 (0.003%) |
| Duration | string | Flight duration | 0 |
| Direct | string | Direct flight indicator | 0 |
| Transit | string | Transit information | 39,756 (24.4%) |
| Baggage | string | Baggage allowance | 160,522 (98.6%) |
| Airline | string | Airline name | 5 (0.003%) |
| Airfare(NZ$) | float | Ticket price in NZ dollars | 0 |

## Data Quality Issues Identified

### 1. Missing Values
- **Baggage**: 98.58% missing (160,522 records)
- **Transit**: 24.42% missing (39,756 records)
- **Other columns**: Minor missing values

### 2. Data Type Issues
- Date and time columns stored as strings
- Inconsistent string formatting

### 3. Outliers
- **Airfare**: 3,121 outliers detected and handled

## Data Cleaning Process

### 1. Missing Value Treatment
- **Numerical columns**: Filled with median values
- **Categorical columns**: Filled with mode values

### 2. Data Type Conversion
- `Travel Date`: Converted to datetime format
- `Dep. time` & `Arr. time`: Converted to time format

### 3. Outlier Handling
- Used IQR method for outlier detection
- Clipped outliers to reasonable bounds: [-35.50, 840.50]

### 4. String Cleaning
- Removed leading/trailing whitespace
- Standardized case formatting

## Files Description

### Input Files
- `NZ airfares.csv`: Original raw dataset

### Output Files
- `cleaned_NZ_airfares.csv`: Cleaned dataset ready for analysis
- `data_cleaning_analysis.png`: Visualization of data quality analysis

### Code Files
- `ml.py`: Main data cleaning script

## Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Running the Data Cleaning
```bash
python ml.py
```

### Expected Output
1. Console output showing:
   - Basic data information
   - Missing value analysis
   - Data type information
   - Cleaning progress
   - Data quality report

2. Generated files:
   - `cleaned_NZ_airfares.csv`
   - `data_cleaning_analysis.png`

## Data Analysis Results

### Key Statistics
- **Average Airfare**: NZ$ 411.03
- **Price Range**: NZ$ 32 - NZ$ 1,364
- **Median Price**: NZ$ 392.00

### Data Completeness
- **Before cleaning**: 9.09%
- **After cleaning**: 100%

### Airlines in Dataset
- Air New Zealand
- Jetstar
- And others...

## Visualization Features

The script generates a comprehensive visualization including:
1. **Missing Values Heatmap**: Shows distribution of missing data
2. **Numerical Columns Boxplot**: Displays outlier distribution
3. **Categorical Value Distribution**: Shows frequency of categories
4. **Correlation Heatmap**: Displays relationships between numerical variables

## Data Quality Metrics

| Metric | Value |
|--------|-------|
| Original Records | 162,833 |
| Cleaned Records | 162,833 |
| Missing Values Before | 200,336 |
| Missing Values After | 0 |
| Outliers Detected | 3,121 |
| Data Completeness | 100% |

## Next Steps

This cleaned dataset is now ready for:
1. **Exploratory Data Analysis (EDA)**
2. **Feature Engineering**
3. **Machine Learning Model Development**
4. **Price Prediction Analysis**

## Technical Details

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **scipy**: Statistical functions

### Data Cleaning Methods
- **IQR Method**: For outlier detection
- **Median/Mode Imputation**: For missing value handling
- **String Standardization**: For text data cleaning

## Contact

For questions or issues related to this data cleaning project, please refer to the code comments or create an issue in the repository.

---

**Note**: This dataset contains New Zealand domestic flight information and may be useful for airfare prediction, market analysis, or transportation research projects.