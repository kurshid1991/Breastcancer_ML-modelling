# Breast Cancer Classification using Machine Learning

## Overview

Using the Breast Cancer Wisconsin (Diagnostic) Database, we can create a classifier to help diagnose patients and predict the likelihood of breast cancer. In this project, various machine learning techniques will be explored. Specifically, a Support Vector Machine (SVM) model is implemented, achieving an accuracy of 99%.

## Exploratory Data Analysis (EDA
You can preview thw images in the jupyter file (.ipynb)

### Load the Dataset and Perform Quick Exploratory Analysis

Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset. It helps to identify patterns, detect anomalies, and gather insights before applying machine learning models. 

We obtain a statistical summary of the numerical columns in the dataset, including:

- Count
- Mean
- Standard Deviation
- Minimum & Maximum Values
- Quartiles

### Data Preprocessing

1. **Encoding the Diagnosis Column:**
   - `M` (Malignant) is encoded as `1`
   - `B` (Benign) is encoded as `0`
2. **Setting the ID Column as the Index:**
   - The ID column is not useful for machine learning, so we drop it.

### Class Distribution

To check the number of benign and malignant cases:

```python
print(data.groupby('diagnosis').size())
```

Output:

```
diagnosis
0    357
1    212
```

From this, we see that the majority of cases are benign (0).

### Data Visualization

#### Density Plots

Density plots help visualize the distribution of data. Most features exhibit a general Gaussian (normal) distribution, which is beneficial for machine learning models that assume normality.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(60,45))
data.plot(kind='density', subplots=True, layout=(7,5), sharex=False, legend=False, fontsize=1)
plt.show()
```

#### Correlation Heatmap

A correlation heatmap is used to visualize relationships between features. Highly correlated features can indicate redundancy and may need to be removed or combined.

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
plt.show()
```

The red around the diagonal suggests strong correlation between attributes. Yellow and green patches indicate moderate correlation, while blue areas show negative correlations.

## Train-Test Split

We split the dataset into predictor variables (X) and the target variable (y), then divide it into training and test sets using an 80-20 split.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Machine Learning Algorithms

Since this is a binary classification problem, several algorithms are tested to determine the best performer.

### **Support Vector Machines (SVM)**
SVM is a supervised learning algorithm that finds the optimal hyperplane separating classes. It is effective in high-dimensional spaces and works well with both linear and non-linear decision boundaries.

### **Classification and Regression Trees (CART)**
CART is a decision tree algorithm that splits the dataset based on feature values. It is easy to interpret but prone to overfitting.

### **Gaussian Naïve Bayes (NB)**
NB is a probabilistic classifier based on Bayes' Theorem, assuming that features are conditionally independent. It is particularly effective for text classification and medical diagnoses.

### **k-Nearest Neighbors (KNN)**
KNN is a non-parametric algorithm that classifies instances based on the majority vote of their k-nearest neighbors. It is simple but computationally expensive for large datasets.

## Cross-Validation

To perform initial testing, we use 10-fold cross-validation. Cross-validation helps evaluate the model's performance by training on different subsets of the dataset and averaging the results.

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

model = SVC()
scores = cross_val_score(model, X_train, y_train, cv=10)
print(f"Mean Accuracy: {scores.mean():.2f}")
```

## Conclusion

This project applies machine learning techniques to classify breast cancer cases. A Support Vector Machine model achieved an accuracy of 99%. Further improvements can be made by optimizing hyperparameters and exploring deep learning techniques.

---

## Installation

To run this project, install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

Run the script to preprocess the dataset, train the model, and evaluate its performance.

```bash
python breast_cancer_classification.py
```

## License

This project is open-source and free to use.

