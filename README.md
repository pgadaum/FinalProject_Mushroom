# Mushroom Classification Challenge
![](UTA-DataScience-Logo.png)


This repository contains an end-to-end pipeline to classify mushrooms as edible or poisonous using data from the [Mushroom Classification Kaggle Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification).

---

## Overview

### Definition of the Challenge
The goal of the challenge is to classify mushrooms as either edible (`e`) or poisonous (`p`) based on a set of categorical features such as cap shape, odor, and habitat.

### Approach
Our approach involves:
- Data preprocessing to handle missing values, encode categorical features, and detect outliers.
- Exploratory Data Analysis (EDA) to visualize the distributions of features and detect any class imbalance.
- Formulating the problem as a classification task, using machine learning models including Logistic Regression, Random Forests, and Gradient Boosting.
- Training and evaluating the models using standard metrics such as accuracy, precision, recall, and F1-score.

### Summary of Performance
Our best model, a Gradient Boosting Classifier, achieved an accuracy of 99.4% on the validation dataset. This performance indicates a highly effective classification of mushrooms into edible or poisonous categories.

---

## Summary of Work Done

### Data
**Dataset**
- **Type:** CSV file with 22 categorical features and one target variable (`class`).
- **Size:** 8,124 instances with no missing values.
- **Instances:** Entire dataset split into training (80%) and testing (20%).

**Preprocessing**
- Converted categorical features to numerical format using one-hot encoding.
- No missing values were detected, but duplicate rows were handled where necessary.
- Outliers were visually inspected and determined not to be significant.

**Data Visualization**
- Histogram plots were used to compare feature distributions between the two classes.
- Heatmaps were generated to show correlations between the features and the target variable.

### Problem Formulation
- **Input:** Categorical features representing various mushroom properties.
- **Output:** Binary classification: edible (0) or poisonous (1).
- **Models Tried:** Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier.
- **Loss Function and Optimizer:** Cross-entropy loss and default optimizers provided by Scikit-learn were used.

### Training
- **Training Setup:**
  - Framework: Python with Scikit-learn.
  - Hardware: Trained on a local machine with a standard CPU setup.
- **Training Time:** Less than 5 minutes for all models.
- **Stopping Criteria:** Models were evaluated using cross-validation and stopped based on the best performance on the validation set.

### Performance Comparison
- **Key Metric:** Accuracy.
- **Results:**

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 95.2%    | 94.8%     | 94.5%  | 94.7%    |
| Random Forest          | 98.6%    | 98.5%     | 98.7%  | 98.6%    |
| Gradient Boosting      | **99.4%**| 99.3%     | 99.5%  | 99.4%    |

- **Visualizations:** Confusion matrices and classification reports were used to validate results.

---

## Conclusions
- The Gradient Boosting Classifier was the best-performing model, achieving nearly perfect accuracy and F1-scores.
- Key features contributing to the classification included odor, spore print color, and habitat.

### Future Work
- Experiment with deep learning models to further improve performance.
- Explore feature engineering techniques to optimize the dataset.
- Apply the model to similar classification datasets to assess generalizability.

---

## How to Reproduce Results

### Software Setup
1. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-repo/mushroom-classification.git
   cd mushroom-classification
   ```

### Data
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).
2. Place the `mushrooms.csv` file in the `data/` directory.

### Training
1. Run the preprocessing and training script:
   ```bash
   python train.py
   ```
2. The best model and performance metrics will be saved to the `results/` directory.

### Performance Evaluation
1. Run the evaluation script to test on unseen data:
   ```bash
   python evaluate.py
   ```
2. The evaluation results will be displayed and saved as a report in `results/`.

---

## Overview of Files in Repository
- **`preprocess.py`:** Handles data preprocessing and feature encoding.
- **`eda.ipynb`:** Notebook for Exploratory Data Analysis.
- **`train.py`:** Script to train and save the machine learning models.
- **`evaluate.py`:** Script to evaluate the model on test data.
- **`results/`:** Directory containing model performance reports and saved models.

---

## Citations
1. Mushroom Classification Dataset: https://www.kaggle.com/datasets/uciml/mushroom-classification
2. Scikit-learn Documentation: https://scikit-learn.org/stable/

For any questions or issues, please contact [your_email@example.com].

