# Flight Price Prediction Project

A comprehensive data analysis project for New Zealand airfare prediction using machine learning techniques.

## 📊 Dataset

- **Source**: NZ airfares.csv
- **Records**: 162,833 flight records
- **Features**: 11 columns including travel dates, airports, airlines, duration, and pricing
- **Target**: Airfare prediction in NZ$

## 🛠️ Project Structure

```
COMP647/
├── lab2.py          # Data cleaning and preprocessing
├── lab3.py          # Data exploration and visualization  
├── lab4.py          # Feature engineering and encoding
├── NZ airfares.csv  # Original dataset
└── requirements.txt # Dependencies
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run analysis scripts**:
   ```bash
   python lab2.py  # Data cleaning
   python lab3.py  # Data exploration
   python lab4.py  # Feature engineering
   ```

## 📈 Key Features

- **Data Cleaning**: Duplicate removal, missing value imputation, outlier detection
- **Exploration**: Monthly price analysis, categorical variable distribution
- **Feature Engineering**: Duration conversion, date processing, categorical encoding
- **Scaling**: Min-max and standard scaling techniques
- **Visualization**: Comprehensive charts and statistical plots

## 📋 Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy
- ydata-profiling, category_encoders
- imbalanced-learn

## 📊 Output Files

- `ProfilingReport.html` - Interactive data profiling report
- Various visualization charts and analysis results

## 🎯 Analysis Results

The project provides insights into:
- Monthly airfare trends and seasonality
- Airport route analysis and pricing patterns
- Feature importance and correlation analysis
- Data quality assessment and cleaning strategies
