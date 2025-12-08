# 📊 Interactive Data Analysis Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A modern, interactive data analysis platform built with Streamlit**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

An all-in-one data analysis platform that seamlessly integrates data loading, preprocessing, exploratory analysis, machine learning, clustering, and association rule mining. Built with a focus on user experience and powerful analytics capabilities.

![Platform Demo](https://via.placeholder.com/800x400/3b82f6/ffffff?text=Data+Analysis+Platform)

## ✨ Features

### 📤 Data Management
- **Multi-format Support**: CSV, Excel (.xlsx, .xls), JSON
- **Smart Parsing**: Automatic delimiter and encoding detection
- **Quality Assessment**: Completeness, uniqueness, and consistency scoring
- **Interactive Cleaning**: Handle missing values, duplicates, and outliers

### 🔍 Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Histograms, box plots, violin plots, bar charts
- **Bivariate Analysis**: Scatter plots, density contour plots
- **Multivariate Analysis**: Scatter matrix, parallel coordinates
- **Correlation Analysis**: Heatmap visualization

### 🤖 Machine Learning
- **Regression**: Linear Regression, KNN Regression, Decision Tree Regression
- **Classification**: Logistic Regression, Naive Bayes, KNN, Decision Trees (CART/ID3/C4.5)
- **Auto Optimization**: Intelligent hyperparameter search
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Model Interpretation**: Feature importance, SHAP value analysis

### 📊 Clustering Analysis
- **K-means Clustering**: Optimal K value detection (Elbow method + Silhouette score)
- **DBSCAN Clustering**: Density-based clustering algorithm
- **Performance Optimized**: Automatic sampling for large datasets
- **Rich Visualizations**: Scatter plots, silhouette analysis

### 📉 Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Variance explanation visualization, 2D projection

### 🔗 Association Rule Mining
- **Apriori Algorithm**: Classic algorithm for frequent itemset mining
- **FP-Growth Algorithm**: Efficient algorithm using FP-tree structure (faster for large datasets)
- **Visualizations**: Rule heatmap, Sankey diagram

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd 数据挖掘

# 2. Create Conda environment (recommended)
conda create -n data-analysis python=3.9
conda activate data-analysis

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
# Windows
run.bat

# Linux/Mac
streamlit run main.py
```

The application will automatically open in your browser at `http://localhost:8501`

### Generate Sample Data (Optional)

```bash
python generate_datasets.py
```

## 📁 Project Structure

```
.
├── main.py                    # Streamlit main application
├── data_loader.py            # Data loading and preprocessing
├── visualization.py           # Data visualization (EDA)
├── ml_models.py              # ML unified interface
├── ml_visualization.py       # ML visualizations
├── generate_datasets.py      # Generate sample datasets
├── page_modules/             # Page modules
│   ├── data_pages.py        # Data-related pages
│   └── ml_pages.py          # Machine learning pages
├── utils/                     # Utility modules
│   └── config.py            # Configuration and styles
├── 算法/                      # Algorithm implementations (modularized)
│   ├── utils.py             # Common algorithm utilities
│   ├── 数据准备.py           # Data preparation
│   ├── 回归/                 # Regression algorithms
│   ├── 分类/                 # Classification algorithms
│   ├── 聚类/                 # Clustering algorithms
│   ├── 降维/                 # Dimensionality reduction
│   ├── 关联规则/             # Association rules
│   └── 模型解释/             # Model interpretation
├── data/                     # Sample datasets (auto-generated)
├── background.jpg            # Background image (optional)
├── requirements.txt          # Dependencies
├── environment.yml           # Conda environment config
└── run.bat                   # Windows launcher
```

## 🎨 Interface Highlights

- **Modern UI Design**: Gradient headers, card-based layout, elegant styling
- **Background Support**: Customizable background images
- **Responsive Layout**: Adapts to different screen sizes
- **Interactive Charts**: Plotly-based interactive visualizations

## 📚 Usage Guide

### 1. Data Upload
Upload your data files via drag-and-drop or file selection. Supports CSV with custom delimiters and automatic encoding detection.

### 2. Data Overview
View data statistics, column information, and quality scores at a glance.

### 3. Data Cleaning
- **Missing Values**: Keep, remove, or fill (mean/median/mode)
- **Duplicates**: Remove duplicate rows
- **Outliers**: Detect using Z-score or IQR method

### 4. Exploratory Analysis
Choose from various chart types for comprehensive data exploration across univariate, bivariate, and multivariate analysis.

### 5. Machine Learning
Select task type (classification/regression), choose target and features, pick algorithms, enable auto-optimization, and view results with rich visualizations.

### 6. Clustering Analysis
Choose between K-means or DBSCAN, find optimal K values, and visualize clustering results with silhouette analysis.

## 🔧 Tech Stack

| Category | Technology |
|----------|-----------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Plotly Express |
| Association Rules | MLxtend |
| Model Interpretation | SHAP |
| UI Components | streamlit-option-menu |

## 📝 Dependencies

Key dependencies (see `requirements.txt` for complete list):

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
mlxtend>=0.22.0
shap>=0.42.0
streamlit-option-menu>=0.3.6
```

## 🎯 Algorithms

All algorithms are modularized in the `算法/` directory for easy maintenance and extension.

### Regression
- **Linear Regression**: Least squares method
- **KNN Regression**: Distance-based regression
- **Decision Tree Regression**: CART algorithm

### Classification
- **Logistic Regression**: Maximum likelihood estimation
- **Naive Bayes**: Bayesian theorem-based
- **KNN Classification**: Distance-based classification
- **Decision Tree Classification**: Supports CART/ID3/C4.5

### Clustering
- **K-means**: Partition-based clustering with optimal K detection
- **DBSCAN**: Density-based clustering

### Association Rules
- **Apriori**: Classic frequent pattern mining
- **FP-Growth**: Efficient FP-tree based mining (recommended for large datasets)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Special thanks to the open-source community:
- [Streamlit](https://streamlit.io/) - The framework that makes it all possible
- [Scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
- [Plotly](https://plotly.com/) - Interactive visualizations
- And all other contributors to the libraries we use

## 🐛 Issues

Found a bug or have a feature request? Please [open an issue](https://github.com/yourusername/repo/issues).

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

Made with ❤️ by the Data Analysis Team

</div>
