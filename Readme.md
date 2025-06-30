# 🚀 Startup Success Prediction ML Pipeline

A comprehensive machine learning pipeline for predicting startup success using advanced ML techniques, feature engineering, and hyperparameter optimization.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete end-to-end machine learning pipeline for startup success prediction. The pipeline processes raw startup data through multiple stages including data preprocessing, feature engineering, model building, and hyperparameter optimization.

**Key Highlights:**
- 📊 **Dataset**: 923 startups with 49 features → optimized to 20 features
- 🏆 **Best Model**: XGBoost with 84.26% precision
- ⚡ **Pipeline Time**: ~4 minutes for complete execution
- 🎯 **Business Focus**: High precision for investment decisions

## ✨ Features

### 🧹 Data Preprocessing
- Intelligent missing value handling with business logic
- Date feature engineering and age categorization
- Data quality filtering and outlier removal
- Class balance analysis and validation

### 🔧 Feature Engineering
- **Business Logic Features**: Funding progression indicators
- **Age Categories**: Startup lifecycle stages
- **Investment Patterns**: VC/Angel backing analysis
- **Geographic Features**: State-based location encoding

### 🤖 Model Building
- **5 Advanced Models**: Logistic Regression, Decision Tree, XGBoost, CatBoost, AdaBoost
- **Cross-Validation**: Stratified 5-fold validation
- **Performance Metrics**: ROC-AUC, Precision, Recall, F1-Score
- **Statistical Testing**: Paired t-tests for model comparison

### ⚡ Hyperparameter Optimization
- **Business Constraints**: Minimum precision/recall requirements
- **Grid Search**: Comprehensive parameter exploration
- **Constraint-Aware**: Only selects models meeting business criteria
- **Time Tracking**: Performance monitoring for each optimization

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- Git

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/urilevy77/startups-success-prediction.git
   cd startups-success-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```python
   import pandas as pd
   import sklearn
   import xgboost
   import catboost
   print("✅ All dependencies installed successfully!")
   ```

## 🚀 Quick Start

### Basic Usage
```python
from startups import run_startup_prediction_pipeline

# Run complete pipeline
pipeline, best_model, summary = run_startup_prediction_pipeline(
    data_path='path/to/startup_data.csv'
)

# Get results
print(summary)
```

### Advanced Usage
```python
from startups import StartupMLPipeline

# Initialize pipeline
pipeline = StartupMLPipeline(target_column='labels', random_state=42)

# Custom business constraints
constraints = {
    'min_precision': 0.80,
    'min_recall': 0.70,
    'min_f1': 0.75
}

# Run with custom settings
results = pipeline.run_complete_pipeline(
    raw_data_path='startup_data.csv',
    business_constraints=constraints,
    evaluation_method='precision'
)

# Get best model
best_model = pipeline.get_best_model()
```

## 📊 Pipeline Stages

### Stage 1: Data Preprocessing
```
📁 Raw Data: 923 samples × 49 features
    ↓
🧹 Preprocessing:
    • Missing value analysis and handling
    • Date feature engineering
    • Age categorization
    • Data quality filtering
    ↓
📊 Clean Data: 840 samples × 48 features
```

### Stage 2: Feature Engineering & Selection
```
🔧 Feature Engineering:
    • Business logic features (5 new features)
    • Funding progression indicators
    • Investment pattern analysis
    ↓
🎯 Feature Selection:
    • XGBoost-based importance ranking
    • Cross-validated stability testing
    • Median threshold selection
    ↓
✅ Selected Features: 20 optimal features
```

### Stage 3: Model Building & Evaluation
```
🤖 Model Training:
    • Logistic Regression (baseline)
    • Decision Tree (simple non-linear)
    • XGBoost (gradient boosting)
    • CatBoost (category-optimized)
    • AdaBoost (adaptive boosting)
    ↓
📈 Evaluation:
    • ROC-AUC scoring
    • Cross-validation
    • Statistical testing
```

### Stage 4: Hyperparameter Optimization
```
⚡ Optimization:
    • Business constraint validation
    • Grid search with custom scoring
    • Performance vs efficiency analysis
    ↓
🏆 Best Model: Constraint-aware selection
```

## 📈 Model Performance

### Final Results
| Model | Precision | Recall | F1-Score | ROC-AUC | Constraints Met |
|-------|-----------|--------|----------|---------|-----------------|
| **XGBoost** | **0.8426** | **0.9021** | **0.8713** | **0.8468** | ✅ |
| CatBoost | 0.8332 | 0.9039 | 0.8670 | 0.8513 | ✅ |
| AdaBoost | 0.8221 | 0.9040 | 0.8610 | 0.8459 | ✅ |
| Logistic Regression | 0.8276 | 0.8949 | 0.8598 | 0.8421 | ✅ |
| Decision Tree | 0.8038 | 0.8424 | 0.8223 | 0.7967 | ✅ |

### Statistical Significance
- **XGBoost vs others**: Statistically significant improvements in most comparisons
- **Confidence Level**: 95% (α = 0.05)
- **Validation Method**: Paired t-tests with 10-fold cross-validation

## 💡 Usage Examples

### 1. Run Complete Pipeline
```python
# Simple execution
pipeline, model, summary = run_startup_prediction_pipeline(
    data_path='startup_data.csv'
)
```

### 2. Custom Model Configuration
```python
from sklearn.ensemble import RandomForestClassifier

custom_models = {
    'Random_Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(n_estimators=200)
}

pipeline = StartupMLPipeline()
results = pipeline.run_complete_pipeline(
    raw_data_path='data.csv',
    custom_models=custom_models
)
```

### 3. Individual Stage Execution
```python
# Run only preprocessing
df_clean = preprocessing(raw_df)

# Run feature engineering
df_features, selected_features, target = feature_engineering_selection(df_clean)

# Run model building
trained_models = model_building(df_features, selected_features, target)
```

## 📁 Project Structure
```
startups-success-prediction/
│
├── startups.py                 # Main pipeline implementation
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── data/
│   └── startup_data.csv       # Dataset (if included)
├── results/
│   ├── model_performance.png  # Performance visualizations
│   └── feature_importance.png # Feature analysis plots
└── notebooks/
    ├── pipeline_demo.ipynb    # Jupyter notebook demo
    └── analysis.ipynb         # Detailed analysis
```

## 🎯 Results Summary

### Business Impact
- **High Precision**: 84.26% precision reduces false positive investments
- **Strong Recall**: 90.21% recall captures most successful startups  
- **Balanced Performance**: F1-score of 87.13% provides optimal balance
- **Fast Execution**: Complete pipeline runs in ~4 minutes

### Key Insights
1. **Relationship features** are most predictive (14% importance)
2. **Milestone timing** strongly indicates success
3. **Geographic location** (MA, TX, NY) influences outcomes
4. **Business logic features** prove their predictive value

### Performance Optimization
- **50% feature reduction** with no performance loss
- **Constraint-aware optimization** ensures business requirements
- **Statistical validation** confirms model significance

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Uri Levy** - [GitHub](https://github.com/urilevy77)

Project Link: [https://github.com/urilevy77/startups-success-prediction](https://github.com/urilevy77/startups-success-prediction)

---

## 🙏 Acknowledgments

- Dataset source and original research
- Open source machine learning community
- Contributors and reviewers

---

**⭐ If you found this project helpful, please give it a star!**
