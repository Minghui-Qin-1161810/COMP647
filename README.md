# Flight Price Prediction Project - Assignment 2

A comprehensive data analysis project for New Zealand airfare prediction using machine learning techniques.

## 📊 Dataset

- **Source**: NZ airfares.csv
- **Records**: 162,833 flight records
- **Features**: 11 columns including travel dates, airports, airlines, duration, and pricing
- **Target**: Airfare prediction in NZ$

## 🛠️ Project Structure

```
COMP647/
├── Assignment-2.py     # Comprehensive data analysis and preprocessing
├── NZ airfares.csv     # Original dataset
├── requirements.txt    # Dependencies
├── ProfilingReport.html # Interactive data profiling report
└── ProfilingReport.json # JSON format profiling report
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete analysis**:
   ```bash
   python Assignment-2.py
   ```

## 📈 Analysis Components

### Part 1: Data Cleaning and Preprocessing
- **Duplicate Handling**: Identification and removal of duplicate records
- **Missing Value Treatment**: Multiple strategies for handling missing data
- **Outlier Detection**: IQR and Z-Score methods for outlier identification
- **Data Profiling**: Automated comprehensive data quality report

### Part 2: Data Exploration and Visualization
- **Basic Statistics**: Dataset overview and descriptive statistics
- **Monthly Analysis**: Seasonal price trends and patterns
- **Categorical Analysis**: Airport, airline, and route distribution analysis
- **Visualization**: Interactive charts and statistical plots

### Part 3: Feature Engineering and Encoding
- **Feature Engineering**: 
  - Duration conversion from string to numerical hours
  - Date processing to extract month information
  - Duration categorization into intervals
- **Encoding Techniques**:
  - Label Encoding for categorical variables
  - One-Hot Encoding for binary representation
  - Binary Encoding for efficient storage
- **Data Scaling**: Min-Max and Standard scaling methods
- **Imbalanced Data Handling**: Oversampling techniques demonstration

## 🔧 Key Features

- **Comprehensive Analysis**: All three lab components integrated into a single script
- **Feature Engineering**: Advanced data transformation techniques
- **Multiple Encoding Methods**: Various approaches for categorical data
- **Scaling Techniques**: Data normalization for machine learning
- **Interactive Reports**: HTML-based data profiling with detailed insights
- **Visualization**: Rich charts and statistical plots

## 📋 Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy
- ydata-profiling, category_encoders
- Standard Python libraries

## 📊 Output Files

- `ProfilingReport.html` - Interactive data profiling report
- `ProfilingReport.json` - JSON format for programmatic access
- Various visualization charts and analysis results

## 🎯 Analysis Results

The project provides insights into:
- **Monthly Trends**: Seasonal airfare patterns and price fluctuations
- **Route Analysis**: Airport-specific pricing and demand patterns
- **Feature Importance**: Correlation analysis and feature relationships
- **Data Quality**: Comprehensive assessment and cleaning strategies
- **Feature Engineering**: Transformation techniques for enhanced analysis

## 💡 Reflection

The project demonstrates the importance of feature engineering in data science. By converting non-numerical parameters into numerical ones, we were able to unlock the analytical potential of the dataset and enable comprehensive statistical analysis, including correlation analysis, scaling operations, and machine learning model training.

This will generate all visualizations, reports, and analysis results in a single execution.