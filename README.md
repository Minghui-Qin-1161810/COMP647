# Flight Price Prediction Project - COMP647

A comprehensive machine learning project for New Zealand airfare prediction, covering data analysis, preprocessing, feature engineering, and model development.

## 📊 Dataset

- **Source**: `NZ airfares.csv`
- **Records**: 162,833 flight records
- **Features**: Travel dates, airports, airlines, duration, pricing
- **Target**: Airfare prediction in NZ$

## 🛠️ Project Structure

```
COMP647/
├── Assignment-2.py          # Data cleaning, preprocessing, and feature engineering
├── Assignment-3.py          # Machine learning models, evaluation, and XAI
├── NZ airfares.csv          # Dataset
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Assignment 2** (Data Analysis):
   ```bash
   python Assignment-2.py
   ```

3. **Run Assignment 3** (Machine Learning):
   ```bash
   python Assignment-3.py
   ```

## 📋 Assignment 2: Data Analysis and Preprocessing

### Features
- **Data Cleaning**: Duplicate removal, missing value handling, outlier detection (IQR & Z-Score)
- **Data Exploration**: Statistical analysis, monthly trends, categorical analysis
- **Feature Engineering**: Duration conversion, date processing, categorization
- **Encoding**: Label, One-Hot, and Binary encoding
- **Data Scaling**: Min-Max and Standard scaling
- **Data Profiling**: Automated quality reports

## 🤖 Assignment 3: Machine Learning

### Features
- **Feature Engineering**: Date/time extraction, duration conversion, categorical encoding
- **Feature Selection**: Correlation analysis, F-statistic, Random Forest importance
- **Supervised Learning**: Linear Regression, Random Forest, Gradient Boosting
- **Performance Evaluation**: RMSE, MAE, R² score with detailed analysis
- **Overfitting Prevention**: Train-test split, cross-validation
- **Unsupervised Learning**: K-Means clustering with elbow method
- **Explainable AI**: Feature importance, Random Forest importance, SHAP values


## 📝 Notes
- All code includes comprehensive comments and explanations
- Results are saved as PNG images and HTML reports
- Both assignments are self-contained and can be run independently

## 👤 Author
**Minghui Qin**  
Student ID: 1161810
