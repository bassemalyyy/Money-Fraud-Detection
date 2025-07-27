# Money Fraud Detection

This repository contains a machine learning project focused on detecting fraudulent transactions. The goal is to build and evaluate various classification models to accurately identify fraudulent activities within a financial dataset.

## Project Overview

The project follows a standard machine learning pipeline, including:

1. **Data Importing and Preprocessing**: Loading the dataset, handling missing values, and removing duplicates to ensure data quality.

2. **Feature Engineering**: Creating new features and encoding categorical variables to enhance model performance.

3. **Model Training and Evaluation**: Implementing and comparing several machine learning models, including ensemble methods and traditional classifiers, along with a Neural Network.

4. **Regularization**: Applying regularization techniques to prevent overfitting and improve model generalization.

5. **Performance Comparison**: Evaluating models based on accuracy, precision, recall, F1-score, and ROC AUC curves.

## Dataset

The project utilizes a dataset (assumed to be `fraudData.csv`) containing various transaction details such as `step`, `customer`, `age`, `gender`, `zipcodeOri`, `merchant`, `zipMerchant`, `category`, `amount`, and a `fraud` label.

## Key Steps and Techniques

### Data Preprocessing & Feature Engineering

* **Handling Missing Values and Duplicates**: Initial cleaning to ensure data integrity.

* **Categorical Encoding**: `LabelEncoder` is used for `customer`, `gender`, `merchant`, and `category` columns.

* **Feature Creation**:

  * `log_amount`: Log transformation of the `amount` to handle skewed distributions.

  * `age_amount_interaction`: Interaction term between `age` and `amount`.

  * `hour_of_day` and `day_of_week`: Time-based features extracted from the `step` column.

* **Feature Scaling**: `StandardScaler` is applied to numerical features for models sensitive to feature scales (SVM, Logistic Regression, Neural Network).

### Machine Learning Models

The following classification models are implemented and evaluated:

* **Ensemble Methods**:

  * **Random Forest Classifier**: A powerful ensemble method robust to overfitting. Regularization is explored by limiting `max_depth`, `min_samples_split`, and `min_samples_leaf`.

  * **Gradient Boosting Classifier**: Builds trees sequentially, with each new tree correcting errors of previous ones. Regularization includes `max_depth`, `learning_rate`, `n_estimators`, and `subsample`.

  * **XGBoost Classifier**: An optimized distributed gradient boosting library. Regularization includes `max_depth`, `learning_rate`, `n_estimators`, `reg_alpha` (L1), and `reg_lambda` (L2).

* **Traditional ML Models**:

  * **Logistic Regression**: A linear model for binary classification. L1 and L2 regularization are applied.

  * **Decision Tree Classifier**: A simple, interpretable model. Regularization is applied using `max_depth`, `min_samples_split`, and `min_samples_leaf`.

  * **Support Vector Machine (SVM) - LinearSVC**: A powerful classifier for high-dimensional data. Regularization is controlled by the `C` parameter.

* **Neural Network (ANN)**:

  * A sequential model built with `tf.keras.models.Sequential`.

  * Comprises `Dense` layers with `relu` activation for hidden layers and `sigmoid` for the output layer.

  * Compiled with `adam` optimizer and `binary_crossentropy` loss.

  * Regularization is implemented using `Dropout` layers.

## Results and Comparison

Each model's performance is assessed using:

* **Accuracy Score**

* **Classification Report**: Providing Precision, Recall, and F1-score for both classes.

* **ROC AUC Curve**: Visualizing the trade-off between True Positive Rate and False Positive Rate across different thresholds, enabling a comprehensive comparison of model performance.

The notebook includes detailed output for each model's training and evaluation, showcasing their effectiveness in identifying fraudulent transactions.

## How to Run

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name

2. **Ensure you have the dataset:** Place *fraudData.csv* in the root directory of the project, or update the *file_path* variable in the notebook to its correct location.

3. **Install dependencies:**

```bash
pip install pandas scikit-learn xgboost tensorflow matplotlib seaborn
```

4. **Open and run the Jupyter Notebook.**
